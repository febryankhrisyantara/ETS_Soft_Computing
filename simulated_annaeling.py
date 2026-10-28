import copy
import math
import random
import time
import pandas as pd

# ==========================================
# 1. MEMBACA & PRA-PEMROSESAN DATASET
# ==========================================
df_dosen = pd.read_csv("dataset - dosen.csv")
df_ruangan = pd.read_csv("dataset - ruangan.csv")
df_slot = pd.read_csv("dataset - slot_waktu.csv")
df_mk = pd.read_csv("dataset - mata_kuliah.csv")

list_ruangan = df_ruangan["id_ruangan"].tolist()
list_slot = df_slot["id_slot"].tolist()

data_mk = []
for _, row in df_mk.iterrows():
  dosen_pengajar = [d.strip() for d in str(row["dosen"]).split(",")]
  data_mk.append({
      "id_mk": row["id_mk"],
      "nama_mk": row["nama_mk"],
      "dosen": dosen_pengajar,
  })


# ==========================================
# 2. FUNGSI UTILITAS & PENALTI
# ==========================================
def create_initial_schedule(courses):
  """Membangkitkan jadwal acak awal sebagai titik berangkat."""
  schedule = []
  for mk in courses:
    schedule.append({
        "id_mk": mk["id_mk"],
        "nama_mk": mk["nama_mk"],
        "dosen": mk["dosen"],
        "ruangan": random.choice(list_ruangan),
        "slot_waktu": random.choice(list_slot),
    })
  return schedule


def calculate_penalty(schedule):
  """Menghitung total bentrok (ruangan & dosen) sebagai nilai energi/biaya."""
  penalty = 0
  n = len(schedule)
  for i in range(n):
    for j in range(i + 1, n):
      if schedule[i]["slot_waktu"] == schedule[j]["slot_waktu"]:
        if schedule[i]["ruangan"] == schedule[j]["ruangan"]:
          penalty += 1
        if not set(schedule[i]["dosen"]).isdisjoint(set(schedule[j]["dosen"])):
          penalty += 1
  return penalty


def get_neighbor(schedule):
  """Membuat tetangga baru dengan memodifikasi secara acak satu mata kuliah."""
  neighbor = copy.deepcopy(schedule)
  idx = random.randint(0, len(neighbor) - 1)
  # Pilih acak apakah ingin mengubah ruangan atau slot waktu
  if random.random() < 0.5:
    neighbor[idx]["ruangan"] = random.choice(list_ruangan)
  else:
    neighbor[idx]["slot_waktu"] = random.choice(list_slot)
  return neighbor


# ==========================================
# 3. ALGORITMA SIMULATED ANNEALING
# ==========================================
def simulated_annealing(
    courses, initial_temp=500.0, cooling_rate=0.85, max_iter=5000
):
  current_schedule = create_initial_schedule(courses)
  current_penalty = calculate_penalty(current_schedule)

  best_schedule = copy.deepcopy(current_schedule)
  best_penalty = current_penalty

  temp = initial_temp

  print("Memulai proses Simulated Annealing...")
  for iteration in range(max_iter):
    if best_penalty == 0:
      break

    # Buat solusi tetangga
    neighbor = get_neighbor(current_schedule)
    neighbor_penalty = calculate_penalty(neighbor)

    # Selisih energi (penalti)
    delta_e = neighbor_penalty - current_penalty

    # Jika tetangga lebih baik, atau diterima berdasarkan probabilitas suhu
    if delta_e < 0 or random.random() < math.exp(-delta_e / temp):
      current_schedule = neighbor
      current_penalty = neighbor_penalty

      # Perbarui solusi terbaik global
      if current_penalty < best_penalty:
        best_schedule = copy.deepcopy(current_schedule)
        best_penalty = current_penalty
        print(
            f"Iterasi {iteration:04d} | Suhu: {temp:.2f} | Penalti turun"
            f" menjadi: {best_penalty}"
        )

    # Turunkan suhu
    temp *= cooling_rate

  return best_schedule, best_penalty


# ==========================================
# 4. EKSEKUSI & HASIL
# ==========================================
start = time.perf_counter()
best_schedule, total_penalty = simulated_annealing(data_mk)
elapsed = time.perf_counter() - start

fitness = 1.0 / (1.0 + total_penalty)

print("\n" + "=" * 50)
print("HASIL AKHIR SIMULATED ANNEALING")
print("=" * 50)
print(f"Sisa Bentrok: {total_penalty} | Fitness: {fitness:.4f}")
print(f"Running time SA: {elapsed:.4f} detik")

# Merapikan dataframe hasil
df_hasil = pd.DataFrame(best_schedule)
df_hasil["dosen"] = df_hasil["dosen"].apply(lambda x: ", ".join(x))
df_hasil = df_hasil.merge(
    df_ruangan[["id_ruangan", "nama_ruangan"]],
    left_on="ruangan",
    right_on="id_ruangan",
    how="left",
)
df_hasil = df_hasil.merge(
    df_slot, left_on="slot_waktu", right_on="id_slot", how="left"
)
df_hasil = df_hasil[
    ["id_mk", "nama_mk", "dosen", "nama_ruangan", "hari", "mulai", "selesai"]
]
df_hasil = df_hasil.sort_values(by=["hari", "mulai", "nama_ruangan"])

print("\nJadwal Optimal:")
print(df_hasil.to_string(index=False))