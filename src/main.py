"""Main entry point for the support triage agent."""

import os
import sys
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load .env from the current directory or parent directory
load_dotenv()
# If running from src/, try loading from parent
if not os.getenv("GEMINI_API_KEY"):
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from triage_agent import TriageAgent


def setup_logging(log_file: str = "logs/log.txt"):
    """Setup logging configuration."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    """Main function to run the triage agent."""
    parser = argparse.ArgumentParser(
        description='Multi-Domain Support Triage Agent'
    )
    parser.add_argument(
        '--input', '-i',
        default='data/support_tickets.csv',
        help='Input CSV file with support tickets'
    )
    parser.add_argument(
        '--output', '-o',
        default='output/predictions.csv',
        help='Output CSV file for predictions'
    )
    parser.add_argument(
        '--corpus', '-c',
        default='data/corpus',
        help='Directory containing support corpus'
    )
    parser.add_argument(
        '--api-key',
        default=None,
        help='Gemini API key (or set GEMINI_API_KEY env var)'
    )
    parser.add_argument(
        '--no-embeddings',
        action='store_true',
        help='Disable embedding-based retrieval'
    )
    parser.add_argument(
        '--log-file',
        default='logs/log.txt',
        help='Log file path'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    logger.info("=" * 60)
    logger.info("Multi-Domain Support Triage Agent")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Corpus directory: {args.corpus}")
    
    # Get API key
    api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
    if not api_key:
        logger.warning("No API key provided. Will use fallback response generation.")
    
    # Initialize agent
    logger.info("Initializing triage agent...")
    agent = TriageAgent(
        api_key=api_key,
        corpus_dir=args.corpus,
        use_embeddings=not args.no_embeddings
    )
    
    # Process tickets
    logger.info("Processing tickets...")
    results = agent.process_tickets(args.input, args.output)
    
    # Summary
    status_counts = {}
    for r in results:
        status = r['status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    logger.info("=" * 60)
    logger.info("Processing Complete")
    logger.info("=" * 60)
    logger.info(f"Total tickets processed: {len(results)}")
    for status, count in status_counts.items():
        logger.info(f"  {status}: {count}")
    logger.info(f"Output saved to: {args.output}")
    logger.info(f"Log saved to: {args.log_file}")
    logger.info(f"Finished at: {datetime.now().isoformat()}")


if __name__ == '__main__':
    main()