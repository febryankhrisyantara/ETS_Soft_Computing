import pandas as pd
import random
import copy
import time

# ==========================================
# 1. MEMBACA & PRA-PEMROSESAN DATASET
# ==========================================
# Membaca file CSV
df_dosen = pd.read_csv("dataset - dosen.csv")
df_ruangan = pd.read_csv("dataset - ruangan.csv")
df_slot = pd.read_csv("dataset - slot_waktu.csv")
df_mk = pd.read_csv("dataset - mata_kuliah.csv")

# Ekstraksi ID untuk randomisasi
list_ruangan = df_ruangan['id_ruangan'].tolist()
list_slot = df_slot['id_slot'].tolist()

# Preprocessing: Memisahkan multi-dosen pada mata kuliah
data_mk = []
for index, row in df_mk.iterrows():
    # Split teks dosen berdasarkan koma dan hilangkan spasi ekstra
    dosen_pengajar = [d.strip() for d in str(row['dosen']).split(',')]
    data_mk.append({
        "id_mk": row['id_mk'],
        "nama_mk": row['nama_mk'],
        "dosen": dosen_pengajar
    })

# ==========================================
# 2. TAHAP 1: INISIALISASI POPULASI
# ==========================================
def create_chromosome():
    """Membuat 1 jadwal acak untuk seluruh mata kuliah."""
    schedule = []
    for mk in data_mk:
        schedule.append({
            "id_mk": mk["id_mk"],
            "nama_mk": mk["nama_mk"],
            "dosen": mk["dosen"],
            "ruangan": random.choice(list_ruangan),
            "slot_waktu": random.choice(list_slot)
        })
    return schedule

# ==========================================
# 3. TAHAP 2: EVALUASI (FITNESS)
# ==========================================
def calculate_fitness(chromosome):
    """Menghitung penalti jadwal. Mengembalikan (skor fitness, jumlah penalti)"""
    penalty = 0
    total_genes = len(chromosome)
    
    for i in range(total_genes):
        for j in range(i + 1, total_genes):
            # Jika dua kelas berjalan pada waktu yang sama
            if chromosome[i]["slot_waktu"] == chromosome[j]["slot_waktu"]:
                
                # Cek Bentrok Ruangan
                if chromosome[i]["ruangan"] == chromosome[j]["ruangan"]:
                    penalty += 1
                
                # Cek Bentrok Dosen (Intersection list dosen)
                dosen_i = set(chromosome[i]["dosen"])
                dosen_j = set(chromosome[j]["dosen"])
                if not dosen_i.isdisjoint(dosen_j): 
                    penalty += 1
                    
    # Fitness formula: 1 / (1 + penalti). Sempurna jika = 1.0
    return 1.0 / (1.0 + penalty), penalty

# ==========================================
# 4. TAHAP 4 & 5: CROSSOVER & MUTASI
# ==========================================
def crossover(parent1, parent2):
    """Menggabungkan separuh jadwal Parent 1 & 2"""
    split_point = len(parent1) // 2
    child = parent1[:split_point] + parent2[split_point:]
    return copy.deepcopy(child)

def mutate(chromosome, mutation_rate=0.1):
    """Mengganti ruangan atau waktu secara acak pada beberapa mata kuliah"""
    for gene in chromosome:
        if random.random() < mutation_rate:
            gene["ruangan"] = random.choice(list_ruangan)
            gene["slot_waktu"] = random.choice(list_slot)
    return chromosome

# ==========================================
# 5. TAHAP 3 & 6: MAIN LOOP (SELEKSI & ITERASI)
# ==========================================
POPULATION_SIZE = 50
GENERATIONS = 200
MUTATION_RATE = 0.1

# Mulai hitung waktu eksekusi (hanya algoritma, tanpa baca CSV & output)
start_time = time.perf_counter()

# Membuat populasi awal
population = [create_chromosome() for _ in range(POPULATION_SIZE)]

print("="*50)
print("MEMULAI PENCARIAN JADWAL OPTIMAL")
print("="*50)

best_schedule = None
history_penalty = float('inf')

for gen in range(GENERATIONS):
    # Evaluasi & Urutkan populasi (terbaik di index 0)
    population = sorted(population, key=lambda x: calculate_fitness(x)[0], reverse=True)
    
    best_fitness, current_penalty = calculate_fitness(population[0])
    
    # Log Pelacakan: Cetak jika menemukan kombinasi yang lebih baik
    if current_penalty < history_penalty:
        print(f"[Generasi {gen:03d}] Ditemukan jadwal lebih baik! | Sisa Bentrok: {current_penalty} | Fitness: {best_fitness:.4f}")
        history_penalty = current_penalty
    
    # Hentikan pencarian jika tidak ada yang bentrok (Fitness == 1.0)
    if current_penalty == 0:
        best_schedule = population[0]
        print(f"\n>>> JADWAL SEMPURNA DITEMUKAN PADA GENERASI {gen} <<<")
        break
        
    # Seleksi (Elitism): Membawa 10 jadwal terbaik langsung ke generasi selanjutnya
    next_generation = population[:10]
    
    # Melengkapi sisa populasi dengan Reproduksi (Crossover + Mutasi)
    while len(next_generation) < POPULATION_SIZE:
        # Seleksi Turnamen (Memilih 2 induk dari top 50% populasi)
        p1 = random.choice(population[:POPULATION_SIZE//2])
        p2 = random.choice(population[:POPULATION_SIZE//2])
        
        child = crossover(p1, p2)
        child = mutate(child, MUTATION_RATE)
        next_generation.append(child)
        
    population = next_generation

# Jika loop selesai tapi belum sempurna, ambil yang terbaik
if history_penalty > 0:
    best_schedule = population[0]
    print(f"\nBatas generasi tercapai. Menampilkan jadwal terbaik dengan sisa bentrok: {history_penalty}")

elapsed_time = time.perf_counter() - start_time
print(f"Running time GA: {elapsed_time:.4f} detik")

# ==========================================
# 6. MENAMPILKAN DAN MENYIMPAN HASIL
# ==========================================
# Konversi jadwal terbaik kembali ke dalam DataFrame
df_hasil = pd.DataFrame(best_schedule)
df_hasil['dosen'] = df_hasil['dosen'].apply(lambda x: ', '.join(x)) # Gabungkan kembali nama dosen

# Merge dengan file CSV lainnya untuk mendapatkan nama ruangan, hari, dan jam
df_hasil = df_hasil.merge(df_ruangan[['id_ruangan', 'nama_ruangan']], left_on='ruangan', right_on='id_ruangan', how='left')
df_hasil = df_hasil.merge(df_slot, left_on='slot_waktu', right_on='id_slot', how='left')

# Rapikan urutan kolom agar mudah dibaca
df_hasil = df_hasil[['id_mk', 'nama_mk', 'dosen', 'nama_ruangan', 'hari', 'mulai', 'selesai']]
df_hasil = df_hasil.sort_values(by=['hari', 'mulai', 'nama_ruangan'])

print("\n" + "="*70)
print("HASIL AKHIR PENJADWALAN:")
print("="*70)
print(df_hasil.to_string(index=False))

# (Opsional) Simpan ke CSV jika dibutuhkan
# df_hasil.to_csv("jadwal_optimal_terbaru.csv", index=False)