import sys
import time
import threading

class Spinner:
    def __init__(self, total_files):
        self.stop_event = threading.Event()
        self.spinner_thread = threading.Thread(target=self._spinner_task)
        self.total_files = total_files
        self.start_time = time.time()

    def _spinner_task(self):
        spinner = ['|', '/', '-', '\\']
        idx = 0
        while not self.stop_event.is_set():
            sys.stdout.write(f"\r{spinner[idx % len(spinner)]} Processing: ")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.1)

    def start(self):
        self.spinner_thread.start()

    def stop(self):
        self.stop_event.set()
        self.spinner_thread.join()

    def update_progress(self, current_file, filename):
        elapsed_time = time.time() - self.start_time
        avg_time_per_file = elapsed_time / (current_file + 1)
        remaining_time = avg_time_per_file * (self.total_files - (current_file + 1))
        progress = (current_file + 1) / self.total_files * 100

        # Update progress information
        progress_bar = f"[{int(progress // 2) * '='}{(50 - int(progress // 2)) * ' '}]"
        sys.stdout.write(f"\r  Processing: {filename} ({current_file + 1}/{self.total_files}) {progress_bar} {progress:.2f}% | Elapsed: {elapsed_time:.2f}s | Remaining: {remaining_time:.2f}s")
        sys.stdout.flush()