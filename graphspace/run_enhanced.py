#!/usr/bin/env python3
import os
import sys
from enhanced_graphspace import EnhancedGraphSpace


def main():
    """Main entry point for the enhanced GraphSpace web application."""
    # Ensure directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("config", exist_ok=True)

    # Create GraphSpace instance
    try:
        graphspace = EnhancedGraphSpace(
            data_path="data/user_data.json",
            config_path="config/config.json",
            use_api=True,
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            use_google_drive=False
        )

        # Import web app and run it
        try:
            from app.app import run_app
            run_app(graphspace)
        except ImportError as e:
            print(f"Error loading web interface: {e}")
            sys.exit(1)

    except Exception as e:
        print(f"\nError initializing EnhancedGraphSpace: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
