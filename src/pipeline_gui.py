import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import threading
import sys
import os
from datetime import datetime, timezone, timedelta

class PipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Arbitrage Pipeline Launcher")
        self.root.geometry("750x650")
        
        # Target Date Frame
        date_frame = ttk.LabelFrame(root, text="Target Date (Format: YYYY-MM-DD)", padding=10)
        date_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.date_var = tk.StringVar()
        default_date = datetime.now(timezone(timedelta(hours=5, minutes=30))) - timedelta(days=1)
        self.date_var.set(default_date.strftime("%Y-%m-%d"))
        
        date_entry = ttk.Entry(date_frame, textvariable=self.date_var, width=50)
        date_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.info_label = ttk.Label(root, text="")
        self.info_label.pack(padx=10, pady=5)
        
        self.date_var.trace_add("write", self.update_info)
        self.update_info()
        
        # â”€â”€â”€ Scraper Settings Frame â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        scraper_frame = ttk.LabelFrame(root, text="Scraper Settings", padding=10)
        scraper_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Price Min
        price_row = ttk.Frame(scraper_frame)
        price_row.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(price_row, text="Minimum Price (â‚¹):").pack(side=tk.LEFT)
        self.price_min_var = tk.StringVar(value="25000")
        price_spinbox = ttk.Spinbox(
            price_row, from_=0, to=500000, increment=5000,
            textvariable=self.price_min_var, width=12
        )
        price_spinbox.pack(side=tk.LEFT, padx=(10, 0))
        
        ttk.Label(price_row, text="  (listings below this price are ignored)", 
                  foreground="gray").pack(side=tk.LEFT, padx=(10, 0))
        
        # Warranty Only Checkbox
        warranty_row = ttk.Frame(scraper_frame)
        warranty_row.pack(fill=tk.X, pady=(0, 5))
        
        self.warranty_var = tk.BooleanVar(value=True)
        warranty_check = ttk.Checkbutton(
            warranty_row, text="Warranty records only",
            variable=self.warranty_var, command=self._update_warranty_hint
        )
        warranty_check.pack(side=tk.LEFT)
        
        self.warranty_hint = ttk.Label(
            warranty_row,
            text="  âœ“ Pre-filters for warranty mentions â€” saves LLM tokens",
            foreground="green"
        )
        self.warranty_hint.pack(side=tk.LEFT, padx=(5, 0))
        
        # â”€â”€â”€ Negotiation Margin Frame â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        margin_frame = ttk.LabelFrame(root, text="Negotiation Buffer (%)", padding=10)
        margin_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.margin_var = tk.StringVar(value="0")
        margin_spinbox = ttk.Spinbox(margin_frame, from_=0, to=100, increment=5, textvariable=self.margin_var, width=10)
        margin_spinbox.pack(side=tk.LEFT, padx=(0, 10))
        
        margin_label = ttk.Label(margin_frame, text="Max % above FMV you are willing to negotiate")
        margin_label.pack(side=tk.LEFT)
        
        # Run button
        self.run_btn = ttk.Button(root, text="Run Pipeline", command=self.run_pipeline)
        self.run_btn.pack(pady=10)
        
        # Log Output
        log_frame = ttk.LabelFrame(root, text="Live Execution Logs", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_area = scrolledtext.ScrolledText(log_frame, state=tk.DISABLED, bg="black", fg="lightgreen", font=("Consolas", 10))
        self.log_area.pack(fill=tk.BOTH, expand=True)
        
    def _update_warranty_hint(self):
        if self.warranty_var.get():
            self.warranty_hint.config(
                text="  âœ“ Pre-filters for warranty mentions â€” saves LLM tokens",
                foreground="green"
            )
        else:
            self.warranty_hint.config(
                text="  âš  All laptops scraped â€” higher LLM token cost",
                foreground="orange"
            )
        
    def _get_target_datetime(self):
        val = self.date_var.get().strip()
        try:
            dt = datetime.strptime(val, "%Y-%m-%d")
            return dt.replace(tzinfo=timezone(timedelta(hours=5, minutes=30)))
        except ValueError:
            return None

    def update_info(self, *args):
        dt = self._get_target_datetime()
        if dt:
            self.info_label.config(text=f"Output will be saved as: arbitrage_opportunities_{dt.strftime('%Y-%m-%d')}.csv")
        else:
            self.info_label.config(text="Output will be saved as: (invalid format, use YYYY-MM-DD)")

    def log(self, message):
        self.log_area.config(state=tk.NORMAL)
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.log_area.config(state=tk.DISABLED)
        
    def run_pipeline(self):
        dt = self._get_target_datetime()
        if not dt:
            self.log("[ERROR] Invalid date format. Please use YYYY-MM-DD (e.g., 2026-06-15)\n")
            return
        target_date = dt.isoformat()
        
        margin = self.margin_var.get().strip()
        try:
            float(margin)
        except ValueError:
            margin = "0"
        
        # Validate price_min
        price_min = self.price_min_var.get().strip()
        try:
            int(price_min)
        except ValueError:
            price_min = "25000"
        
        warranty_only = self.warranty_var.get()
            
        self.run_btn.config(state=tk.DISABLED)
        self.log_area.config(state=tk.NORMAL)
        self.log_area.delete(1.0, tk.END)
        self.log_area.config(state=tk.DISABLED)
        self.log(f"Starting pipeline with date: {target_date}")
        self.log(f"  Price min: â‚¹{price_min}  |  Warranty only: {warranty_only}  |  Margin: {margin}%")
        self.log("-" * 60)
        
        def run_thread():
            try:
                # Build command
                cmd = [
                    sys.executable, "master_predictor.py",
                    "--date", target_date,
                    "--margin", margin,
                    "--price-min", price_min,
                ]
                if not warranty_only:
                    cmd.append("--no-warranty-filter")
                
                # Run the master_predictor script
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )
                
                for line in process.stdout:
                    self.root.after(0, self.log, line.rstrip('\n'))
                    
                process.wait()
                self.root.after(0, self.log, f"\n{'-' * 60}\nPipeline finished with exit code {process.returncode}")
            except Exception as e:
                self.root.after(0, self.log, f"\n[FATAL ERROR] {e}")
            finally:
                self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
                
        threading.Thread(target=run_thread, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = PipelineGUI(root)
    root.mainloop()


