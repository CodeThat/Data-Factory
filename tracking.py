import os
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


def configure_logging():
    log_directory = "log"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_file = os.path.join(log_directory, "tracking.log")

    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


class DirectoryChangeHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        # Handle event here (e.g., print event type, file path, etc.)
        logging.info(f"Event type: {event.event_type}, File path: {event.src_path}")


def track_directory_changes(directory_path):
    event_handler = DirectoryChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, directory_path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()


if __name__ == "__main__":
    configure_logging()

    directory_path = "./new_articles"  # Replace with your directory path
    track_directory_changes(directory_path)
