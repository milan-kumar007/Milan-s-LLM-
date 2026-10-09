import os
import sys
import time
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox
import torch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.inference import load_model, sample_next_token

class GPTPlaygroundApp(tk.Tk):
    """
    Modern Desktop GUI for interactive inference and parameter tuning
    of custom trained GPT language models.
    """
    def __init__(self):
        super().__init__()

        self.title("Milan's LLM (124M) Playground - GPU Accelerated")
        self.geometry("1060x780")
        self.minsize(850, 650)

        # State
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.current_checkpoint = ""
        self.is_generating = False
        self.stop_requested = threading.Event()
        self.stream_queue = queue.Queue()

        # UI Styling
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self._configure_styles()
        self._build_ui()
        self._load_available_checkpoints()

        # Poll streaming updates from generation worker
        self.after(20, self._process_stream_queue)

    def _configure_styles(self):
        # Configure fonts and ttk elements
        self.configure(bg="#f4f6f9")
        self.style.configure(".", font=("Segoe UI", 9))
        self.style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), background="#f4f6f9", foreground="#1a2530")
        self.style.configure("SubHeader.TLabel", font=("Segoe UI", 10, "bold"), background="#f4f6f9", foreground="#2c3e50")
        self.style.configure("Param.TLabel", font=("Segoe UI", 9), background="#ffffff", foreground="#34495e")
        self.style.configure("Status.TLabel", font=("Segoe UI", 9), background="#e9ecef", foreground="#2b2d42")
        self.style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

    def _build_ui(self):
        # Main Layout: Top header, Left controls sidebar, Right text playground
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # 1. Header Banner
        header_frame = tk.Frame(self, bg="#1e293b", height=50)
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        header_frame.grid_propagate(False)

        title_lbl = tk.Label(
            header_frame,
            text=" 🧠 Milan's LLM Playground (124M)",
            font=("Segoe UI", 13, "bold"),
            bg="#1e293b",
            fg="#f8fafc"
        )
        title_lbl.pack(side="left", padx=15, pady=8)

        device_desc = f"Device: {self.device.upper()}"
        if self.device == "cuda":
            device_desc += f" ({torch.cuda.get_device_name(0)})"
        self.device_badge = tk.Label(
            header_frame,
            text=device_desc,
            font=("Segoe UI", 9, "bold"),
            bg="#0f766e" if self.device == "cuda" else "#64748b",
            fg="#ffffff",
            padx=10,
            pady=3
        )
        self.device_badge.pack(side="right", padx=15, pady=10)

        # 2. Left Sidebar (Settings & Hyperparameters)
        sidebar = tk.Frame(self, bg="#ffffff", width=320, relief="flat", highlightbackground="#e2e8f0", highlightthickness=1)
        sidebar.grid(row=1, column=0, sticky="ns", padx=(10, 5), pady=10)
        sidebar.grid_propagate(False)

        self._build_sidebar(sidebar)

        # 3. Right Content Area (Prompt & Streaming Output)
        main_content = tk.Frame(self, bg="#f8fafc")
        main_content.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=10)
        main_content.grid_rowconfigure(1, weight=1)
        main_content.grid_columnconfigure(0, weight=1)

        self._build_main_content(main_content)

        # 4. Bottom Status Bar
        status_bar = tk.Frame(self, bg="#e2e8f0", height=28)
        status_bar.grid(row=2, column=0, columnspan=2, sticky="ew")

        self.status_lbl = tk.Label(status_bar, text="Ready", font=("Segoe UI", 9), bg="#e2e8f0", fg="#334155")
        self.status_lbl.pack(side="left", padx=10)

        self.metrics_lbl = tk.Label(
            status_bar,
            text="Tokens: 0 | Speed: 0.0 tok/s | Time: 0.0s",
            font=("Segoe UI", 9, "bold"),
            bg="#e2e8f0",
            fg="#0369a1"
        )
        self.metrics_lbl.pack(side="right", padx=10)

    def _build_sidebar(self, parent):
        pad_x = 14
        pad_y = 6

        # Checkpoint Selection Section
        lbl_ckpt = tk.Label(parent, text="MODEL CHECKPOINT", font=("Segoe UI", 9, "bold"), bg="#ffffff", fg="#64748b")
        lbl_ckpt.pack(anchor="w", padx=pad_x, pady=(12, 4))

        self.combo_ckpt = ttk.Combobox(parent, state="readonly", font=("Segoe UI", 9))
        self.combo_ckpt.pack(fill="x", padx=pad_x, pady=2)
        self.combo_ckpt.bind("<<ComboboxSelected>>", self._on_checkpoint_selected)

        btn_reload_ckpt = ttk.Button(parent, text="🔄 Reload Model", command=self._reload_model_thread)
        btn_reload_ckpt.pack(fill="x", padx=pad_x, pady=(4, 10))

        # Checkpoint Info Badge
        self.ckpt_info_lbl = tk.Label(parent, text="No model loaded", font=("Segoe UI", 8), bg="#f1f5f9", fg="#475569", anchor="w", justify="left", padx=6, pady=4)
        self.ckpt_info_lbl.pack(fill="x", padx=pad_x, pady=(0, 10))

        sep1 = ttk.Separator(parent, orient="horizontal")
        sep1.pack(fill="x", padx=pad_x, pady=6)

        # Sampling Parameters Section
        lbl_params = tk.Label(parent, text="SAMPLING PARAMETERS", font=("Segoe UI", 9, "bold"), bg="#ffffff", fg="#64748b")
        lbl_params.pack(anchor="w", padx=pad_x, pady=(6, 8))

        # Temperature
        self.temp_var = tk.DoubleVar(value=0.8)
        self._create_slider_control(
            parent,
            title="Temperature",
            var=self.temp_var,
            from_=0.1, to_=1.8, resolution=0.05,
            format_str="{:.2f}",
            tooltip="Higher = more creative, Lower = more deterministic"
        )

        # Top-K
        self.top_k_var = tk.IntVar(value=40)
        self._create_slider_control(
            parent,
            title="Top-K Sampling",
            var=self.top_k_var,
            from_=0, to_=100, resolution=1,
            format_str="{:d}",
            tooltip="Number of top probability tokens to sample from (0 = disabled)"
        )

        # Top-P (Nucleus)
        self.top_p_var = tk.DoubleVar(value=0.9)
        self._create_slider_control(
            parent,
            title="Top-P (Nucleus)",
            var=self.top_p_var,
            from_=0.1, to_=1.0, resolution=0.05,
            format_str="{:.2f}",
            tooltip="Cumulative probability threshold (1.0 = disabled)"
        )

        # Repetition Penalty
        self.rep_pen_var = tk.DoubleVar(value=1.1)
        self._create_slider_control(
            parent,
            title="Repetition Penalty",
            var=self.rep_pen_var,
            from_=1.0, to_=2.0, resolution=0.05,
            format_str="{:.2f}",
            tooltip="Penalizes repeated tokens (1.0 = none)"
        )

        # Max New Tokens
        self.max_tokens_var = tk.IntVar(value=100)
        self._create_slider_control(
            parent,
            title="Max New Tokens",
            var=self.max_tokens_var,
            from_=10, to_=400, resolution=10,
            format_str="{:d}",
            tooltip="Maximum length of newly generated tokens"
        )

        # Instruction / Chat Mode Toggle
        self.instruct_mode_var = tk.BooleanVar(value=True)
        chk_instruct = ttk.Checkbutton(
            parent,
            text="Instruction / Q&A Mode (Chat)",
            variable=self.instruct_mode_var
        )
        chk_instruct.pack(anchor="w", padx=pad_x, pady=(4, 6))

        # Preset Prompts Dropdown
        lbl_presets = tk.Label(parent, text="TASK PRESETS", font=("Segoe UI", 9, "bold"), bg="#ffffff", fg="#64748b")
        lbl_presets.pack(anchor="w", padx=pad_x, pady=(4, 4))

        self.presets = [
            "Write an email to my class teacher asking for leave.",
            "What does a GPU do?",
            "What is the difference between a CPU and a GPU?",
            "Explain what an algorithm is.",
            "Give three tips for staying healthy.",
            "Write a Python function to check if a number is prime."
        ]
        self.combo_presets = ttk.Combobox(parent, values=self.presets, state="readonly", font=("Segoe UI", 9))
        self.combo_presets.pack(fill="x", padx=pad_x, pady=2)
        self.combo_presets.bind("<<ComboboxSelected>>", self._on_preset_selected)

    def _create_slider_control(self, parent, title, var, from_, to_, resolution, format_str, tooltip=""):
        frame = tk.Frame(parent, bg="#ffffff")
        frame.pack(fill="x", padx=14, pady=3)

        top_row = tk.Frame(frame, bg="#ffffff")
        top_row.pack(fill="x")

        lbl = tk.Label(top_row, text=title, font=("Segoe UI", 9), bg="#ffffff", fg="#334155")
        lbl.pack(side="left")

        val_lbl = tk.Label(top_row, text=format_str.format(var.get()), font=("Segoe UI", 9, "bold"), bg="#ffffff", fg="#0284c7")
        val_lbl.pack(side="right")

        def on_change(val):
            formatted = format_str.format(var.get())
            val_lbl.config(text=formatted)

        scale = ttk.Scale(
            frame,
            from_=from_,
            to=to_,
            variable=var,
            orient="horizontal",
            command=on_change
        )
        scale.pack(fill="x", pady=(1, 2))

    def _build_main_content(self, parent):
        # 1. Prompt Input Box
        prompt_frame = tk.LabelFrame(parent, text=" Prompt Input ", font=("Segoe UI", 10, "bold"), bg="#f8fafc", fg="#334155", padx=10, pady=6)
        prompt_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=(0, 6))

        self.txt_prompt = tk.Text(prompt_frame, height=3, wrap="word", font=("Consolas", 10), relief="solid", borderwidth=1, bg="#ffffff")
        self.txt_prompt.pack(fill="x", pady=(2, 6))
        self.txt_prompt.insert("1.0", "Write an email to my class teacher asking for leave.")

        # Buttons Bar
        btn_bar = tk.Frame(prompt_frame, bg="#f8fafc")
        btn_bar.pack(fill="x")

        self.btn_generate = tk.Button(
            btn_bar,
            text="▶  Generate Text",
            font=("Segoe UI", 10, "bold"),
            bg="#16a34a",
            fg="#ffffff",
            activebackground="#15803d",
            activeforeground="#ffffff",
            relief="flat",
            padx=14,
            pady=4,
            cursor="hand2",
            command=self._start_generation
        )
        self.btn_generate.pack(side="left", padx=(0, 6))

        self.btn_stop = tk.Button(
            btn_bar,
            text="⏹  Stop",
            font=("Segoe UI", 10, "bold"),
            bg="#dc2626",
            fg="#ffffff",
            activebackground="#b91c1c",
            activeforeground="#ffffff",
            relief="flat",
            padx=12,
            pady=4,
            state="disabled",
            cursor="hand2",
            command=self._stop_generation
        )
        self.btn_stop.pack(side="left", padx=6)

        self.btn_clear = ttk.Button(btn_bar, text="🗑 Clear Output", command=self._clear_output)
        self.btn_clear.pack(side="right", padx=(6, 0))

        self.btn_copy = ttk.Button(btn_bar, text="📋 Copy", command=self._copy_output)
        self.btn_copy.pack(side="right", padx=6)

        # 2. Streaming Output Display
        output_frame = tk.LabelFrame(parent, text=" Real-Time Generated Output ", font=("Segoe UI", 10, "bold"), bg="#f8fafc", fg="#334155", padx=10, pady=8)
        output_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=6)
        output_frame.grid_rowconfigure(0, weight=1)
        output_frame.grid_columnconfigure(0, weight=1)

        scroll = ttk.Scrollbar(output_frame)
        scroll.grid(row=0, column=1, sticky="ns")

        self.txt_output = tk.Text(
            output_frame,
            wrap="word",
            font=("Consolas", 11),
            relief="solid",
            borderwidth=1,
            bg="#ffffff",
            yscrollcommand=scroll.set,
            padx=12,
            pady=10
        )
        self.txt_output.grid(row=0, column=0, sticky="nsew")
        scroll.config(command=self.txt_output.yview)

        # Text Tag Formats
        self.txt_output.tag_configure("prompt_tag", foreground="#475569", font=("Consolas", 11, "bold"))
        self.txt_output.tag_configure("generated_tag", foreground="#0f172a", font=("Consolas", 11))
        self.txt_output.tag_configure("info_tag", foreground="#2563eb", font=("Consolas", 9, "italic"))

    def _load_available_checkpoints(self):
        ckpt_dir = "models/checkpoints"
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)

        files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        def sort_key(name):
            if name == "sft_best_model.pt":
                return (0, name)
            if name == "best_model.pt":
                return (1, name)
            return (2, name)
        files.sort(key=sort_key)

        if files:
            self.combo_ckpt["values"] = files
            self.combo_ckpt.current(0)
            self._load_model_threaded(os.path.join(ckpt_dir, files[0]))
        else:
            self.combo_ckpt["values"] = ["No checkpoints found"]
            self.status_lbl.config(text="No .pt checkpoints found in models/checkpoints")

    def _on_checkpoint_selected(self, event=None):
        selected = self.combo_ckpt.get()
        if selected and selected != "No checkpoints found":
            path = os.path.join("models/checkpoints", selected)
            self._load_model_threaded(path)

    def _reload_model_thread(self):
        selected = self.combo_ckpt.get()
        if selected and selected != "No checkpoints found":
            self._load_model_threaded(os.path.join("models/checkpoints", selected))

    def _load_model_threaded(self, path):
        self.status_lbl.config(text=f"Loading {os.path.basename(path)} on {self.device.upper()}...")
        self.btn_generate.config(state="disabled")

        def worker():
            try:
                mdl, tok, meta = load_model(path, device=self.device)
                self.model = mdl
                self.tokenizer = tok
                self.current_checkpoint = path

                info_str = f"Checkpoint: {os.path.basename(path)}\nEpoch: {meta.get('epoch', '-')}, Step: {meta.get('step', '-')}"
                if meta.get("val_loss") is not None:
                    info_str += f"\nVal Loss: {meta['val_loss']:.4f}"

                def update_ui():
                    self.ckpt_info_lbl.config(text=info_str)
                    self.status_lbl.config(text=f"Loaded {os.path.basename(path)} successfully.")
                    self.btn_generate.config(state="normal")

                self.after(0, update_ui)
            except Exception as e:
                def update_err():
                    self.status_lbl.config(text=f"Error loading checkpoint: {e}")
                    messagebox.showerror("Model Loading Error", str(e))
                self.after(0, update_err)

        threading.Thread(target=worker, daemon=True).start()

    def _on_preset_selected(self, event=None):
        preset = self.combo_presets.get()
        if preset:
            self.txt_prompt.delete("1.0", tk.END)
            self.txt_prompt.insert("1.0", preset)

    def _clear_output(self):
        self.txt_output.delete("1.0", tk.END)
        self.metrics_lbl.config(text="Tokens: 0 | Speed: 0.0 tok/s | Time: 0.0s")

    def _copy_output(self):
        text = self.txt_output.get("1.0", tk.END).strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.status_lbl.config(text="Copied generated text to clipboard.")

    def _start_generation(self):
        if self.model is None or self.tokenizer is None:
            messagebox.showwarning("Model Not Loaded", "Please select a valid checkpoint first.")
            return

        prompt = self.txt_prompt.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Empty Prompt", "Please enter a prompt to start generation.")
            return

        self.is_generating = True
        self.stop_requested.clear()
        self.btn_generate.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.status_lbl.config(text="Generating response in real-time...")

        is_instruct = self.instruct_mode_var.get()
        # Setup Output Display
        self.txt_output.delete("1.0", tk.END)
        if is_instruct:
            self.txt_output.insert(tk.END, f"Task: {prompt}\n\nAnswer:\n", "prompt_tag")
        else:
            self.txt_output.insert(tk.END, prompt, "prompt_tag")
        self.txt_output.see(tk.END)

        # Collect hyperparameters
        params = {
            "temperature": float(self.temp_var.get()),
            "top_k": int(self.top_k_var.get()),
            "top_p": float(self.top_p_var.get()),
            "repetition_penalty": float(self.rep_pen_var.get()),
            "max_new_tokens": int(self.max_tokens_var.get()),
            "is_instruct": is_instruct
        }

        # Launch Worker Thread
        thread = threading.Thread(
            target=self._generation_worker,
            args=(prompt, params),
            daemon=True
        )
        thread.start()

    def _generation_worker(self, user_prompt, params):
        model = self.model
        tokenizer = self.tokenizer
        device = self.device

        if params.get("is_instruct", False):
            formatted_prompt = f"### Instruction:\n{user_prompt}\n\n### Response:\n"
        else:
            formatted_prompt = user_prompt

        encoded = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        if not encoded:
            encoded = [tokenizer.eos_token_id]

        tokens = list(encoded)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        eos_id = tokenizer.eos_token_id

        use_bf16 = (device == "cuda" and torch.cuda.is_bf16_supported())
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

        generated_count = 0
        start_time = time.time()

        for _ in range(params["max_new_tokens"]):
            if self.stop_requested.is_set():
                break

            # Slicing context window
            if input_tensor.size(1) > model.config.block_size:
                idx_cond = input_tensor[:, -model.config.block_size:]
            else:
                idx_cond = input_tensor

            with torch.amp.autocast(device_type=device, dtype=amp_dtype):
                logits, _ = model(idx_cond)

            next_logits = logits[:, -1, :]

            next_token_id = sample_next_token(
                next_logits,
                context_tokens=tokens,
                temperature=params["temperature"],
                top_k=params["top_k"],
                top_p=params["top_p"],
                repetition_penalty=params["repetition_penalty"]
            )

            if next_token_id == eos_id:
                break

            tokens.append(next_token_id)
            input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
            token_str = tokenizer.decode([next_token_id], errors="replace")

            if "###" in token_str or "<|endoftext|>" in token_str:
                break

            generated_count += 1
            elapsed = time.time() - start_time
            tok_per_sec = generated_count / max(0.001, elapsed)

            # Send update to main thread queue
            self.stream_queue.put({
                "type": "token",
                "text": token_str,
                "count": generated_count,
                "speed": tok_per_sec,
                "elapsed": elapsed
            })

            # Small yield to maintain UI smoothness
            time.sleep(0.008)

        total_elapsed = time.time() - start_time
        final_speed = generated_count / max(0.001, total_elapsed)
        self.stream_queue.put({
            "type": "done",
            "count": generated_count,
            "speed": final_speed,
            "elapsed": total_elapsed,
            "stopped": self.stop_requested.is_set()
        })

    def _process_stream_queue(self):
        """Processes token messages delivered by the background worker."""
        while not self.stream_queue.empty():
            msg = self.stream_queue.get_nowait()
            mtype = msg.get("type")

            if mtype == "token":
                self.txt_output.insert(tk.END, msg["text"], "generated_tag")
                self.txt_output.see(tk.END)
                self.metrics_lbl.config(
                    text=f"Tokens: {msg['count']} | Speed: {msg['speed']:.1f} tok/s | Time: {msg['elapsed']:.1f}s"
                )

            elif mtype == "done":
                self.is_generating = False
                self.btn_generate.config(state="normal")
                self.btn_stop.config(state="disabled")

                status_text = "Generation stopped by user." if msg.get("stopped") else "Generation completed."
                self.status_lbl.config(text=status_text)
                self.metrics_lbl.config(
                    text=f"Done: {msg['count']} tokens | Avg: {msg['speed']:.1f} tok/s | Total: {msg['elapsed']:.1f}s"
                )

        self.after(20, self._process_stream_queue)

    def _stop_generation(self):
        if self.is_generating:
            self.stop_requested.set()
            self.status_lbl.config(text="Stopping generation...")

if __name__ == "__main__":
    app = GPTPlaygroundApp()
    app.mainloop()
