"""
The Splendid ops app — the AppSheet app the crews run jobs from.

AppSheet reads its jobs from a Postgres database. This module creates the job
there by posting to the Apps Script webhook that already owns that database
(the same one the AppSheet bots call for dispatch, stage and payment messages).
The webhook:

  - checks the shared secret,
  - finds the customer by phone (last 10 digits) or email, or creates them,
  - inserts the job, keyed on the agent's job fingerprint, so a retried request
    returns the job it already made instead of making a second one.

Why not AppSheet's own API or a direct database connection: the database is on
Cloud SQL with an IP allow-list, Railway's outbound IPs change, and the AppSheet
API cannot match customers by normalised phone. The webhook already has the
database connection, the matching rules and the logging, so it is the one door.

Respects DRY_RUN like every other write.
"""

import logging
from typing import Any

import requests
from langsmith import traceable

from services import config

logger = logging.getLogger(__name__)

#: The webhook geocodes nothing and sends nothing on CREATE_JOB, but Apps Script
#: cold starts plus a Cloud SQL connection can take several seconds.
_TIMEOUT = 45


class JobsAppError(RuntimeError):
    """The ops app refused or failed to create the job. The message says why."""


@traceable(run_type="tool", name="jobs_app.create_job")
def create_job(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Create (or find, on a retry) the job in the ops app.

    Returns {"job_id": str, "customer_id": str, "created": bool, "dry_run": bool}.
    `created` is False when the fingerprint already existed — a retry of a
    request whose response was lost, not a new job.
    """
    if config.dry_run():
        logger.info(
            "[DRY RUN] jobs_app.create_job %s on %s",
            payload.get("full_name"), payload.get("move_date"),
        )
        return {"job_id": "dry-run-job-id", "customer_id": "dry-run-customer-id",
                "created": True, "dry_run": True}

    body = {
        **payload,
        "secret": config.jobs_app_webhook_secret(),
        "event_type": "CREATE_JOB",
    }

    # Apps Script answers a POST to /exec with a 302 to the result page.
    # requests follows it as a GET, which is exactly what Google expects; the
    # job has already been written by the time the redirect is issued.
    resp = requests.post(config.jobs_app_webhook_url(), json=body, timeout=_TIMEOUT)

    if not resp.ok:
        raise JobsAppError(f"Ops app webhook returned HTTP {resp.status_code}: {resp.text[:300]}")

    try:
        data = resp.json()
    except ValueError:
        # Usually an HTML Google sign-in page: the URL is not the /exec URL, or
        # the deployment is not set to "Anyone" can access.
        raise JobsAppError(
            "Ops app webhook did not return JSON — check JOBS_APP_WEBHOOK_URL is the "
            "deployment's /exec URL and the deployment is accessible to Anyone. "
            f"Got: {resp.text[:200]!r}"
        ) from None

    if data.get("status") != "success":
        # The webhook always answers 200 (Apps Script cannot set a status code),
        # so a rejection — wrong secret, missing field — arrives here.
        raise JobsAppError(f"Ops app refused the job: {data.get('message') or data}")

    job_id = data.get("job_id")
    if not job_id:
        raise JobsAppError(f"Ops app reported success without a job id: {data}")

    return {
        "job_id": job_id,
        "customer_id": data.get("customer_id", ""),
        "created": bool(data.get("created", True)),
        "dry_run": False,
    }
