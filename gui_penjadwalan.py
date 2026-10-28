import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import random
import copy
import threading
import time
from collections import Counter

ALGO_GA = "Genetic Algorithm"
ALGO_GREEDY = "Greedy Algorithm"


class JadwalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Penjadwalan - ITS Soft Computing")
        self.root.geometry("950x680")

        # Variabel Pilihan Algoritma
        self.algo_var = tk.StringVar(value=ALGO_GA)

        # Variabel Parameter GA
        self.pop_size_var = tk.IntVar(value=50)
        self.gen_var = tk.IntVar(value=200)
        self.mut_rate_var = tk.DoubleVar(value=0.1)

        # Variabel Penampung Hasil
        self.df_hasil_terakhir = None
        self.runtime_var = tk.StringVar(value="Running time: -")

        self.create_widgets()
        self.on_algo_change()

    def create_widgets(self):
        frame_kontrol = tk.LabelFrame(self.root, text="Konfigurasi Algoritma & Kontrol", padx=10, pady=10)
        frame_kontrol.pack(fill="x", padx=10, pady=5)

        tk.Label(frame_kontrol, text="Algoritma:").grid(row=0, column=0, sticky="w")
        self.cmb_algo = ttk.Combobox(frame_kontrol, textvariable=self.algo_var, state="readonly",
                                     values=[ALGO_GA, ALGO_GREEDY], width=18)
        self.cmb_algo.grid(row=0, column=1, columnspan=2, padx=5, sticky="w")
        self.cmb_algo.bind("<<ComboboxSelected>>", self.on_algo_change)

        self.btn_run = tk.Button(frame_kontrol, text="Mulai", bg="green", fg="white", font=("Arial", 9, "bold"), command=self.start_thread)
        self.btn_run.grid(row=0, column=6, padx=10)

        # export csv
        self.btn_export = tk.Button(frame_kontrol, text="Ekspor CSV", bg="green", fg="white", font=("Arial", 9, "bold"), state="disabled", command=self.export_csv)
        self.btn_export.grid(row=0, column=7, padx=5)

        # Parameter GA (dinonaktifkan saat Greedy dipilih)
        self.ga_widgets = []
        tk.Label(frame_kontrol, text="Ukuran Populasi:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        e1 = tk.Entry(frame_kontrol, textvariable=self.pop_size_var, width=8)
        e1.grid(row=1, column=1, padx=5, pady=(8, 0))

        tk.Label(frame_kontrol, text="Maksimal Generasi:").grid(row=1, column=2, sticky="w", padx=10, pady=(8, 0))
        e2 = tk.Entry(frame_kontrol, textvariable=self.gen_var, width=8)
        e2.grid(row=1, column=3, padx=5, pady=(8, 0))

        tk.Label(frame_kontrol, text="Mutation Rate:").grid(row=1, column=4, sticky="w", padx=10, pady=(8, 0))
        e3 = tk.Entry(frame_kontrol, textvariable=self.mut_rate_var, width=8)
        e3.grid(row=1, column=5, padx=5, pady=(8, 0))
        self.ga_widgets = [e1, e2, e3]

        frame_log = tk.LabelFrame(self.root, text="Live Monitoring (Log)", padx=10, pady=10)
        frame_log.pack(fill="x", padx=10, pady=5)

        self.log_text = tk.Text(frame_log, height=8, bg="black", fg="lime", font=("Consolas", 10))
        self.log_text.pack(fill="x")

        self.lbl_runtime = tk.Label(self.root, textvariable=self.runtime_var, font=("Arial", 10, "bold"), anchor="w")
        self.lbl_runtime.pack(fill="x", padx=12)

        frame_tabel = tk.LabelFrame(self.root, text="Hasil Penjadwalan Akhir", padx=10, pady=10)
        frame_tabel.pack(fill="both", expand=True, padx=10, pady=5)

        kolom = ("ID MK", "Mata Kuliah", "Dosen", "Ruangan", "Hari", "Mulai", "Selesai")
        self.tree = ttk.Treeview(frame_tabel, columns=kolom, show="headings")

        for col in kolom:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor="center")

        self.tree.column("Mata Kuliah", width=180, anchor="w")
        self.tree.column("Dosen", width=180, anchor="w")

        scrollbar = ttk.Scrollbar(frame_tabel, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)

    def on_algo_change(self, _event=None):
        """Parameter GA tidak relevan untuk Greedy, jadi dinonaktifkan agar tidak membingungkan."""
        state = "normal" if self.algo_var.get() == ALGO_GA else "disabled"
        for w in self.ga_widgets:
            w.config(state=state)

    def write_log(self, message):
        """Fungsi untuk menulis log ke kotak teks. Dari thread lain, panggil via root.after."""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def log(self, message):
        """Thread-safe logger untuk worker thread."""
        self.root.after(0, self.write_log, message)

    def start_thread(self):
        """Memulai proses di thread terpisah agar GUI tidak freeze (Not Responding)"""
        algo = self.algo_var.get()
        self.btn_run.config(state="disabled", text="Memproses...")
        self.btn_export.config(state="disabled")
        self.cmb_algo.config(state="disabled")
        self.log_text.delete(1.0, tk.END)
        self.df_hasil_terakhir = None
        self.runtime_var.set(f"Running time ({algo}): menghitung...")

        # clear previous data
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Ambil nilai Tk variable di main thread (tidak aman dibaca dari worker)
        try:
            params = (self.pop_size_var.get(), self.gen_var.get(), self.mut_rate_var.get())
        except tk.TclError:
            messagebox.showerror("Error", "Parameter GA harus berupa angka yang valid!")
            self.enable_controls()
            self.runtime_var.set("Running time: -")
            return

        thread = threading.Thread(target=self.run_algorithm, args=(algo, params), daemon=True)
        thread.start()

    # ------------------------------------------------------------------
    # ALGORITMA
    # ------------------------------------------------------------------
    @staticmethod
    def penalty_of(schedule):
        """Hitung jumlah bentrok ruangan & dosen (metrik sama dengan main.py)."""
        pen = 0
        n = len(schedule)
        for i in range(n):
            for j in range(i + 1, n):
                if schedule[i]["slot_waktu"] == schedule[j]["slot_waktu"]:
                    if schedule[i]["ruangan"] == schedule[j]["ruangan"]:
                        pen += 1
                    if not set(schedule[i]["dosen"]).isdisjoint(schedule[j]["dosen"]):
                        pen += 1
        return pen

    def solve_ga(self, data_mk, list_ruangan, list_slot, params):
        """Genetic Algorithm. Mengembalikan jadwal terbaik."""
        POP_SIZE, GENS, MUT_RATE = params

        def create_chrom():
            return [{"id_mk": mk["id_mk"], "nama_mk": mk["nama_mk"], "dosen": mk["dosen"],
                     "ruangan": random.choice(list_ruangan), "slot_waktu": random.choice(list_slot)}
                    for mk in data_mk]

        def fitness(chrom):
            pen = self.penalty_of(chrom)
            return 1.0 / (1.0 + pen), pen

        self.log(f"[INFO] Membangkitkan {POP_SIZE} populasi awal...")
        population = [create_chrom() for _ in range(POP_SIZE)]

        history_penalty = float('inf')

        for gen in range(GENS):
            population = sorted(population, key=lambda x: fitness(x)[0], reverse=True)
            best_fit, current_pen = fitness(population[0])

            if current_pen < history_penalty:
                self.log(f"[Generasi {gen:03d}] Penalti Bentrok turun menjadi: {current_pen} (Fitness: {best_fit:.4f})")
                history_penalty = current_pen

            if current_pen == 0:
                self.log(f">>> Jadwal optimal ditemukan pada generasi ke-{gen} <<<")
                return population[0]

            # reproduksi
            next_gen = population[:10]
            while len(next_gen) < POP_SIZE:
                p1 = random.choice(population[:POP_SIZE // 2])
                p2 = random.choice(population[:POP_SIZE // 2])

                sp = len(p1) // 2
                child = copy.deepcopy(p1[:sp] + p2[sp:])

                for gene in child:
                    if random.random() < MUT_RATE:
                        gene["ruangan"] = random.choice(list_ruangan)
                        gene["slot_waktu"] = random.choice(list_slot)
                next_gen.append(child)
            population = next_gen

        self.log(f"Batas generasi habis. Sisa bentrok terendah: {history_penalty}")
        return population[0]

    def solve_greedy(self, data_mk, list_ruangan, list_slot):
        """Greedy: MK tersulit dulu, tiap MK ditaruh di (slot, ruangan) berpenalti terkecil."""
        load = Counter(d for mk in data_mk for d in mk["dosen"])
        ordered = sorted(
            data_mk,
            key=lambda mk: (len(mk["dosen"]), sum(load[d] for d in mk["dosen"])),
            reverse=True,
        )

        used_rooms = set()   # {(slot, ruangan)}
        used_dosen = set()   # {(slot, dosen)}
        schedule = []

        self.log(f"[INFO] Greedy menjadwalkan {len(ordered)} mata kuliah...")
        for mk in ordered:
            best = None  # (penalty, slot, room)
            for slot in list_slot:
                for room in list_ruangan:
                    p = (1 if (slot, room) in used_rooms else 0) + \
                        sum(1 for d in mk["dosen"] if (slot, d) in used_dosen)
                    if best is None or p < best[0]:
                        best = (p, slot, room)
                    if p == 0:
                        break
                if best[0] == 0:
                    break

            _, slot, room = best
            used_rooms.add((slot, room))
            used_dosen.update((slot, d) for d in mk["dosen"])
            schedule.append({"id_mk": mk["id_mk"], "nama_mk": mk["nama_mk"], "dosen": mk["dosen"],
                             "ruangan": room, "slot_waktu": slot})
        return schedule

    def run_algorithm(self, algo, params):
        try:
            self.log("[INFO] Membaca dataset CSV...")
            df_ruangan = pd.read_csv("dataset - ruangan.csv")
            df_slot = pd.read_csv("dataset - slot_waktu.csv")
            df_mk = pd.read_csv("dataset - mata_kuliah.csv")

            list_ruangan = df_ruangan['id_ruangan'].tolist()
            list_slot = df_slot['id_slot'].tolist()

            data_mk = []
            for _, row in df_mk.iterrows():
                dosen_pengajar = [d.strip() for d in str(row['dosen']).split(',')]
                data_mk.append({
                    "id_mk": row['id_mk'], "nama_mk": row['nama_mk'], "dosen": dosen_pengajar
                })

            # Timer hanya mencakup algoritma (tanpa I/O CSV & render tabel)
            start = time.perf_counter()
            if algo == ALGO_GREEDY:
                best_schedule = self.solve_greedy(data_mk, list_ruangan, list_slot)
            else:
                best_schedule = self.solve_ga(data_mk, list_ruangan, list_slot, params)
            elapsed = time.perf_counter() - start

            penalty = self.penalty_of(best_schedule)
            self.log(f"[HASIL] {algo} | Sisa bentrok: {penalty} | Running time: {elapsed:.4f} detik")
            self.root.after(0, self.runtime_var.set,
                            f"Running time ({algo}): {elapsed:.4f} detik  |  Sisa bentrok: {penalty}")

            self.log("[INFO] Merapikan dan memasukkan data ke tabel...")
            df_hasil = pd.DataFrame(best_schedule)
            df_hasil['dosen'] = df_hasil['dosen'].apply(lambda x: ', '.join(x))
            df_hasil = df_hasil.merge(df_ruangan[['id_ruangan', 'nama_ruangan']], left_on='ruangan', right_on='id_ruangan', how='left')
            df_hasil = df_hasil.merge(df_slot, left_on='slot_waktu', right_on='id_slot', how='left')

            df_hasil = df_hasil[['id_mk', 'nama_mk', 'dosen', 'nama_ruangan', 'hari', 'mulai', 'selesai']]
            df_hasil = df_hasil.sort_values(by=['hari', 'mulai', 'nama_ruangan'])

            self.df_hasil_terakhir = df_hasil

            self.root.after(0, self.update_table, df_hasil)

        except Exception as e:
            self.log(f"[ERROR] Terjadi Kesalahan: {str(e)}")
            self.root.after(0, self.runtime_var.set, "Running time: - (gagal)")

        finally:
            self.root.after(0, self.enable_controls)

    def update_table(self, df_hasil):
        """Memasukkan data hasil ke tabel Treeview"""
        for _, row in df_hasil.iterrows():
            self.tree.insert("", "end", values=list(row))
        messagebox.showinfo("Selesai", "Proses penjadwalan telah selesai!")

    def enable_controls(self):
        """Aktifkan kembali tombol kontrol setelah proses selesai"""
        self.btn_run.config(state="normal", text="Mulai")
        self.cmb_algo.config(state="readonly")
        if self.df_hasil_terakhir is not None:
            self.btn_export.config(state="normal")

    def export_csv(self):
        """Fungsi untuk mengekspor DataFrame hasil ke file CSV"""
        if self.df_hasil_terakhir is None or self.df_hasil_terakhir.empty:
            messagebox.showwarning("Peringatan", "Belum ada data jadwal yang dapat diekspor!")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV File", "*.csv"), ("All Files", "*.*")],
            title="Simpan Hasil Penjadwalan"
        )

        if file_path:
            try:
                self.df_hasil_terakhir.to_csv(file_path, index=False)
                messagebox.showinfo("Berhasil", f"Data jadwal berhasil disimpan ke:\n{file_path}")
                self.write_log(f"[INFO] File berhasil diekspor ke: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal menyimpan file: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = JadwalApp(root)
    root.mainloop()
