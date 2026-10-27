import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import random
import copy
import threading
import time

class JadwalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Penjadwalan GA - ITS Soft Computing")
        self.root.geometry("900x600")
        
        # Variabel Parameter
        self.pop_size_var = tk.IntVar(value=50)
        self.gen_var = tk.IntVar(value=200)
        self.mut_rate_var = tk.DoubleVar(value=0.1)
        
        self.create_widgets()
        
    def create_widgets(self):
        # --- FRAME 1: PARAMETER & KONTROL ---
        frame_kontrol = tk.LabelFrame(self.root, text="Konfigurasi Parameter GA", padx=10, pady=10)
        frame_kontrol.pack(fill="x", padx=10, pady=5)
        
        tk.Label(frame_kontrol, text="Ukuran Populasi:").grid(row=0, column=0, sticky="w")
        tk.Entry(frame_kontrol, textvariable=self.pop_size_var, width=10).grid(row=0, column=1, padx=5)
        
        tk.Label(frame_kontrol, text="Maksimal Generasi:").grid(row=0, column=2, sticky="w", padx=10)
        tk.Entry(frame_kontrol, textvariable=self.gen_var, width=10).grid(row=0, column=3, padx=5)
        
        tk.Label(frame_kontrol, text="Mutation Rate:").grid(row=0, column=4, sticky="w", padx=10)
        tk.Entry(frame_kontrol, textvariable=self.mut_rate_var, width=10).grid(row=0, column=5, padx=5)
        
        self.btn_run = tk.Button(frame_kontrol, text="Mulai Pemrosesan GA", bg="green", fg="white", font=("Arial", 10, "bold"), command=self.start_thread)
        self.btn_run.grid(row=0, column=6, padx=20)
        
        # --- FRAME 2: LOG PROSES (LIVE MONITORING) ---
        frame_log = tk.LabelFrame(self.root, text="Live Monitoring (Log Evolusi)", padx=10, pady=10)
        frame_log.pack(fill="x", padx=10, pady=5)
        
        self.log_text = tk.Text(frame_log, height=8, bg="black", fg="lime", font=("Consolas", 10))
        self.log_text.pack(fill="x")
        
        # --- FRAME 3: TABEL HASIL (TREEVIEW) ---
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

    def write_log(self, message):
        """Fungsi untuk menulis log ke kotak teks secara aman dari thread lain"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END) # Auto-scroll ke bawah
        
    def start_thread(self):
        """Memulai proses GA di thread terpisah agar GUI tidak freeze (Not Responding)"""
        self.btn_run.config(state="disabled", text="Sedang Memproses...")
        self.log_text.delete(1.0, tk.END) # Bersihkan log sebelumnya
        
        # Hapus data di tabel jika ada
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        thread = threading.Thread(target=self.run_ga)
        thread.start()
        
    def run_ga(self):
        try:
            # 1. BACA DATA
            self.write_log("[INFO] Membaca dataset CSV...")
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
                
            # GA Parameters dari GUI
            POP_SIZE = self.pop_size_var.get()
            GENS = self.gen_var.get()
            MUT_RATE = self.mut_rate_var.get()
            
            # 2. FUNGSI GA LOKAL
            def create_chrom():
                return [{"id_mk": mk["id_mk"], "nama_mk": mk["nama_mk"], "dosen": mk["dosen"],
                         "ruangan": random.choice(list_ruangan), "slot_waktu": random.choice(list_slot)} 
                        for mk in data_mk]

            def fitness(chrom):
                pen = 0
                n = len(chrom)
                for i in range(n):
                    for j in range(i+1, n):
                        if chrom[i]["slot_waktu"] == chrom[j]["slot_waktu"]:
                            if chrom[i]["ruangan"] == chrom[j]["ruangan"]: pen += 1
                            if not set(chrom[i]["dosen"]).isdisjoint(set(chrom[j]["dosen"])): pen += 1
                return 1.0 / (1.0 + pen), pen

            # 3. PROSES EVOLUSI
            self.write_log(f"[INFO] Membangkitkan {POP_SIZE} populasi awal...")
            population = [create_chrom() for _ in range(POP_SIZE)]
            
            best_schedule = None
            history_penalty = float('inf')
            
            for gen in range(GENS):
                population = sorted(population, key=lambda x: fitness(x)[0], reverse=True)
                best_fit, current_pen = fitness(population[0])
                
                if current_pen < history_penalty:
                    # Tampilkan log di GUI menggunakan metode thread-safe (after)
                    msg = f"[Generasi {gen:03d}] Penalti Bentrok turun menjadi: {current_pen} (Fitness: {best_fit:.4f})"
                    self.root.after(0, self.write_log, msg)
                    history_penalty = current_pen
                    
                if current_pen == 0:
                    best_schedule = population[0]
                    self.root.after(0, self.write_log, f">>> JADWAL SEMPURNA DITEMUKAN PADA GENERASI {gen} <<<")
                    break
                    
                # Reproduksi
                next_gen = population[:10]
                while len(next_gen) < POP_SIZE:
                    p1 = random.choice(population[:POP_SIZE//2])
                    p2 = random.choice(population[:POP_SIZE//2])
                    
                    # Crossover
                    sp = len(p1) // 2
                    child = copy.deepcopy(p1[:sp] + p2[sp:])
                    
                    # Mutate
                    for gene in child:
                        if random.random() < MUT_RATE:
                            gene["ruangan"] = random.choice(list_ruangan)
                            gene["slot_waktu"] = random.choice(list_slot)
                    next_gen.append(child)
                population = next_gen
                time.sleep(0.01) # Memberi nafas sedikit agar GUI sangat responsif
                
            if history_penalty > 0:
                best_schedule = population[0]
                self.root.after(0, self.write_log, f"Batas generasi habis. Sisa bentrok terendah: {history_penalty}")
                
            # 4. TAMPILKAN HASIL KE TABEL GUI
            self.write_log("[INFO] Merapikan dan memasukkan data ke tabel...")
            df_hasil = pd.DataFrame(best_schedule)
            df_hasil['dosen'] = df_hasil['dosen'].apply(lambda x: ', '.join(x))
            df_hasil = df_hasil.merge(df_ruangan[['id_ruangan', 'nama_ruangan']], left_on='ruangan', right_on='id_ruangan', how='left')
            df_hasil = df_hasil.merge(df_slot, left_on='slot_waktu', right_on='id_slot', how='left')
            
            df_hasil = df_hasil[['id_mk', 'nama_mk', 'dosen', 'nama_ruangan', 'hari', 'mulai', 'selesai']]
            df_hasil = df_hasil.sort_values(by=['hari', 'mulai', 'nama_ruangan'])
            
            # Memasukkan baris DataFrame ke Tkinter Treeview secara thread-safe
            self.root.after(0, self.update_table, df_hasil)
            
        except Exception as e:
            self.root.after(0, self.write_log, f"[ERROR] Terjadi Kesalahan: {str(e)}")
            
        finally:
            # Kembalikan tombol ke keadaan semula
            self.root.after(0, lambda: self.btn_run.config(state="normal", text="Mulai Pemrosesan GA"))

    def update_table(self, df_hasil):
        """Memasukkan data hasil ke tabel Treeview"""
        for _, row in df_hasil.iterrows():
            self.tree.insert("", "end", values=list(row))
        messagebox.showinfo("Selesai", "Proses penjadwalan telah selesai!")

# Inisiasi Aplikasi
if __name__ == "__main__":
    root = tk.Tk()
    app = JadwalApp(root)
    root.mainloop()