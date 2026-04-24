"""Main entry point for the application."""

import asyncio
import argparse
import os
from agent.orchestrator import AgenticCopilot
from config import settings
from utils.logging import get_logger

logger = get_logger(__name__)

if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


async def run_scheduler():
    """Run the proactive agent scheduler."""
    copilot = AgenticCopilot()
    logger.info(
        "Starting scheduler with %s minute interval",
        settings.scrape_interval_minutes,
    )

    while True:
        try:
            settings.reload()
            logger.info("Scheduler cycle starting")
            await copilot.run_cycle()
            logger.info("Scheduler cycle completed")
        except Exception as e:
            logger.error(f"Error in agent cycle: {e}")

        # Sleep for configured interval
        interval_seconds = max(settings.scrape_interval_minutes, 1) * 60
        await asyncio.sleep(interval_seconds)


async def run_single_cycle():
    """Run one agent cycle and exit."""
    copilot = AgenticCopilot()
    assessments = await copilot.run_cycle()
    logger.info(f"Completed single cycle with {len(assessments)} assessments")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Agentic AI Supply Chain Copilot")
    parser.add_argument(
        "--mode",
        choices=["scheduler", "single"],
        default="scheduler",
        help="Run mode: continuous scheduler or single cycle",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Clear vector store and start fresh",
    )

    args = parser.parse_args()

    if args.force_refresh:
        from memory.vector_store import VectorStore
        store = VectorStore()
        store.clear()
        logger.info("Vector store cleared")

    if args.mode == "scheduler":
        logger.info("Starting in scheduler mode")
        asyncio.run(run_scheduler())
    else:
        logger.info("Starting in single cycle mode")
        asyncio.run(run_single_cycle())


if __name__ == "__main__":
    main()
