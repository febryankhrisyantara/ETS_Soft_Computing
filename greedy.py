import time
from collections import Counter

import pandas as pd

# ==========================================
# 1. MEMBACA & PRA-PEMROSESAN DATASET
# ==========================================
df_dosen = pd.read_csv("dataset - dosen.csv")
df_ruangan = pd.read_csv("dataset - ruangan.csv")
df_slot = pd.read_csv("dataset - slot_waktu.csv")
df_mk = pd.read_csv("dataset - mata_kuliah.csv")

list_ruangan = df_ruangan['id_ruangan'].tolist()
list_slot = df_slot['id_slot'].tolist()

# Pisahkan multi-dosen pada mata kuliah 
data_mk = []
for _, row in df_mk.iterrows():
    dosen_pengajar = [d.strip() for d in str(row['dosen']).split(',')]
    data_mk.append({
        "id_mk": row['id_mk'],
        "nama_mk": row['nama_mk'],
        "dosen": dosen_pengajar,
    })


# ==========================================
# 2. EVALUASI (FITNESS) & GREEDY ALGORITHM
# ==========================================
def calculate_fitness(schedule):
    """Menghitung total penalti bentrok dan skor fitness jadwal.
    Fitness formula: 1 / (1 + penalti). Sempurna jika = 1.0 (0 bentrok).
    """
    penalty = 0
    total = len(schedule)
    for i in range(total):
        for j in range(i + 1, total):
            if schedule[i]["slot_waktu"] == schedule[j]["slot_waktu"]:
                # Bentrok ruangan
                if schedule[i]["ruangan"] == schedule[j]["ruangan"]:
                    penalty += 1
                # Bentrok dosen
                if not set(schedule[i]["dosen"]).isdisjoint(set(schedule[j]["dosen"])):
                    penalty += 1
    return 1.0 / (1.0 + penalty), penalty


def order_courses(courses):
    """Urutkan MK dari yang paling sulit dijadwalkan (most-constrained-first).
    Why: MK dengan dosen ber-beban tinggi / banyak dosen punya pilihan slot
    paling sedikit, jadi harus ditempatkan sebelum slot habis.
    """
    load = Counter(d for mk in courses for d in mk["dosen"])
    return sorted(
        courses,
        key=lambda mk: (max(load[d] for d in mk["dosen"]), len(mk["dosen"])),
        reverse=True
    )


def local_penalty(slot, room, dosen, used_rooms, used_dosen):
    """Penalti jika MK ditaruh di (slot, room): bentrok ruangan + bentrok dosen.
    """
    penalty = 1 if (slot, room) in used_rooms else 0
    penalty += sum(1 for d in dosen if (slot, d) in used_dosen)
    return penalty


def greedy_schedule(courses, verbose=True):
    """Bangun jadwal: tiap MK dipilih pasangan (slot, ruangan) berpenalti terkecil.
    Pilihan bersifat lokal & final (tanpa backtracking). Tie-break: urutan
    slot lalu ruangan pada dataset, sehingga hasil deterministik.
    Menampilkan progress langkah demi langkah beserta penalti dan fitness.
    """
    used_rooms = set()   # {(slot, ruangan)}
    used_dosen = set()   # {(slot, dosen)}
    schedule = []
    total_courses = len(courses)
    cumulative_penalty = 0

    ordered = order_courses(courses)
    for step, mk in enumerate(ordered, start=1):
        best = None  # (penalty, slot, room)
        for slot in list_slot:
            for room in list_ruangan:
                p = local_penalty(slot, room, mk["dosen"], used_rooms, used_dosen)
                if best is None or p < best[0]:
                    best = (p, slot, room)
                if p == 0:
                    break
            if best[0] == 0:
                break

        p, slot, room = best
        used_rooms.add((slot, room))
        used_dosen.update((slot, d) for d in mk["dosen"])
        schedule.append({
            "id_mk": mk["id_mk"],
            "nama_mk": mk["nama_mk"],
            "dosen": mk["dosen"],
            "ruangan": room,
            "slot_waktu": slot,
        })

        cumulative_penalty += p
        current_fitness = 1.0 / (1.0 + cumulative_penalty)

        if verbose:
            print(f"[Langkah {step:02d}/{total_courses}] {mk['id_mk']} ({mk['nama_mk']}) -> "
                  f"Slot: {slot}, Ruang: {room} | Penalti: {cumulative_penalty} | Fitness: {current_fitness:.4f}")

    if verbose:
        final_fitness, final_penalty = calculate_fitness(schedule)
        print("-" * 70)
        print(f">>> JADWAL SELESAI DISUSUN | Total Penalti: {final_penalty} | Fitness Akhir: {final_fitness:.4f} <<<")
        print("-" * 70)

    return schedule


# ==========================================
# 3. EKSEKUSI
# ==========================================
print("=" * 50)
print("MEMULAI PENJADWALAN DENGAN GREEDY ALGORITHM")
print("=" * 50)

start = time.perf_counter()
best_schedule = greedy_schedule(data_mk)
elapsed = time.perf_counter() - start

print(f"Running time Greedy: {elapsed:.4f} detik")

# ==========================================
# 4. MENAMPILKAN DAN MENYIMPAN HASIL
# ==========================================
df_hasil = pd.DataFrame(best_schedule)
df_hasil['dosen'] = df_hasil['dosen'].apply(lambda x: ', '.join(x))

df_hasil = df_hasil.merge(df_ruangan[['id_ruangan', 'nama_ruangan']], left_on='ruangan', right_on='id_ruangan', how='left')
df_hasil = df_hasil.merge(df_slot, left_on='slot_waktu', right_on='id_slot', how='left')

df_hasil = df_hasil[['id_mk', 'nama_mk', 'dosen', 'nama_ruangan', 'hari', 'mulai', 'selesai']]
df_hasil = df_hasil.sort_values(by=['hari', 'mulai', 'nama_ruangan'])

print("\n" + "=" * 70)
print("HASIL AKHIR PENJADWALAN (GREEDY):")
print("=" * 70)
print(df_hasil.to_string(index=False))

# df_hasil.to_csv("jadwal_greedy.csv", index=False)

#dosen tersibuk
# key=lambda mk: (max(load[d] for d in mk["dosen"]), len(mk["dosen"])),
# reverse=True,

# rata rata beban dosen
# key=lambda mk: sum(load[d] for d in mk["dosen"]) / len(mk["dosen"]),
# reverse=True,