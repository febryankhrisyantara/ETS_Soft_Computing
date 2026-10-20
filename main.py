import pandas as pd
import random
import copy

# ==========================================
# 1. MEMBACA DATASET CSV
# ==========================================
df_dosen = pd.read_csv("dataset - dosen.csv")
df_ruangan = pd.read_csv("dataset - ruangan.csv")
df_slot = pd.read_csv("dataset - slot_waktu.csv")
df_mk = pd.read_csv("dataset - mata_kuliah.csv")

# Mengubah data dataframe ke bentuk list/dictionary agar mudah diproses
list_ruangan = df_ruangan['id_ruangan'].tolist()
list_slot = df_slot['id_slot'].tolist()

# Pra-pemrosesan data mata kuliah (membersihkan nama dosen yang tergabung oleh koma)
data_mk = []
for index, row in df_mk.iterrows():
    # Split nama dosen jika ada lebih dari 1 (misal: "Pak_Izzat, Pak_Dito")
    dosen_pengajar = [d.strip() for d in str(row['dosen']).split(',')]
    data_mk.append({
        "id_mk": row['id_mk'],
        "nama_mk": row['nama_mk'],
        "dosen": dosen_pengajar
    })

# ==========================================
# 2. REPRESENTASI KROMOSOM & INISIALISASI
# ==========================================
def create_chromosome():
    """Membuat 1 jadwal acak untuk seluruh mata kuliah."""
    schedule = []
    for mk in data_mk:
        gene = {
            "id_mk": mk["id_mk"],
            "nama_mk": mk["nama_mk"],
            "dosen": mk["dosen"], # List nama dosen
            "ruangan": random.choice(list_ruangan),
            "slot_waktu": random.choice(list_slot)
        }
        schedule.append(gene)
    return schedule

# ==========================================
# 3. FUNGSI FITNESS (EVALUASI BENTROK)
# ==========================================
def calculate_fitness(chromosome):
    """Menghitung penalti jadwal. Semakin kecil penalti, semakin baik."""
    penalty = 0
    total_genes = len(chromosome)
    
    for i in range(total_genes):
        for j in range(i + 1, total_genes):
            # Cek apakah jadwal i dan j berada di waktu yang sama
            if chromosome[i]["slot_waktu"] == chromosome[j]["slot_waktu"]:
                
                # Penalti 1: Bentrok Ruangan
                if chromosome[i]["ruangan"] == chromosome[j]["ruangan"]:
                    penalty += 1
                
                # Penalti 2: Bentrok Dosen
                # Memeriksa irisan (intersection) antara list dosen di kelas i dan kelas j
                dosen_i = set(chromosome[i]["dosen"])
                dosen_j = set(chromosome[j]["dosen"])
                if not dosen_i.isdisjoint(dosen_j): # True jika ada dosen yang sama
                    penalty += 1
                    
    # Fitness formula: 1 / (1 + penalti). Sempurna jika = 1.0 (Penalti = 0)
    return 1.0 / (1.0 + penalty)

# ==========================================
# 4. OPERATOR GENETIKA (MUTASI & CROSSOVER)
# ==========================================
def crossover(parent1, parent2):
    """Single-point crossover: Menggabungkan separuh jadwal Parent 1 & 2"""
    split_point = len(parent1) // 2
    child = parent1[:split_point] + parent2[split_point:]
    return copy.deepcopy(child)

def mutate(chromosome, mutation_rate=0.1):
    """Mutasi: Mengganti ruangan atau waktu secara acak pada beberapa mata kuliah"""
    for gene in chromosome:
        if random.random() < mutation_rate:
            gene["ruangan"] = random.choice(list_ruangan)
            gene["slot_waktu"] = random.choice(list_slot)
    return chromosome

# ==========================================
# 5. MAIN LOOP ALGORITMA GENETIKA
# ==========================================
POPULATION_SIZE = 50
GENERATIONS = 200
MUTATION_RATE = 0.1

# Inisialisasi populasi awal
population = [create_chromosome() for _ in range(POPULATION_SIZE)]

print("Mulai mencari jadwal optimal...")
best_schedule = None

for gen in range(GENERATIONS):
    # Evaluasi fitness untuk semua kromosom
    population = sorted(population, key=calculate_fitness, reverse=True)
    
    best_fitness = calculate_fitness(population[0])
    
    # Hentikan pencarian jika tidak ada yang bentrok (Fitness == 1.0)
    if best_fitness == 1.0:
        best_schedule = population[0]
        print(f"Jadwal optimal DITEMUKAN pada generasi {gen}!")
        break
        
    # Elitism: Membawa 10 jadwal terbaik langsung ke generasi selanjutnya
    next_generation = population[:10]
    
    # Melengkapi sisa populasi dengan Crossover dan Mutasi
    while len(next_generation) < POPULATION_SIZE:
        # Memilih 2 induk dari top 50% populasi (Turnamen sederhana)
        p1 = random.choice(population[:POPULATION_SIZE//2])
        p2 = random.choice(population[:POPULATION_SIZE//2])
        
        # Crossover & Mutasi
        child = crossover(p1, p2)
        child = mutate(child, MUTATION_RATE)
        next_generation.append(child)
        
    population = next_generation

if best_fitness < 1.0:
    best_schedule = population[0]
    print(f"Batas generasi tercapai. Menampilkan jadwal terbaik dengan skor: {best_fitness:.4f}")

# ==========================================
# 6. MENAMPILKAN DAN MENYIMPAN HASIL
# ==========================================
# Konversi jadwal terbaik ke dalam DataFrame untuk dilihat/disimpan
df_hasil = pd.DataFrame(best_schedule)
df_hasil['dosen'] = df_hasil['dosen'].apply(lambda x: ', '.join(x)) # Gabung kembali list dosen

# Merge (Join) dengan file CSV lainnya untuk mendapatkan nama_ruangan dan hari/jam
df_hasil = df_hasil.merge(df_ruangan[['id_ruangan', 'nama_ruangan']], left_on='ruangan', right_on='id_ruangan', how='left')
df_hasil = df_hasil.merge(df_slot, left_on='slot_waktu', right_on='id_slot', how='left')

# Rapihkan urutan kolom agar mudah dibaca
df_hasil = df_hasil[['id_mk', 'nama_mk', 'dosen', 'nama_ruangan', 'hari', 'mulai', 'selesai']]
df_hasil = df_hasil.sort_values(by=['hari', 'mulai', 'nama_ruangan'])

print("\nHasil Penjadwalan:\n")
print(df_hasil.to_string(index=False))

# (Opsional) Simpan ke Excel/CSV
# df_hasil.to_csv("jadwal_optimal.csv", index=False)