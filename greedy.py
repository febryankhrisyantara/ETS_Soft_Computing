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

# Pisahkan multi-dosen pada mata kuliah (sama seperti main.py)
data_mk = []
for _, row in df_mk.iterrows():
    dosen_pengajar = [d.strip() for d in str(row['dosen']).split(',')]
    data_mk.append({
        "id_mk": row['id_mk'],
        "nama_mk": row['nama_mk'],
        "dosen": dosen_pengajar,
    })


# ==========================================
# 2. GREEDY ALGORITHM
# ==========================================
def order_courses(courses):
    """Urutkan MK dari yang paling sulit dijadwalkan (most-constrained-first).

    Why: MK dengan dosen ber-beban tinggi / banyak dosen punya pilihan slot
    paling sedikit, jadi harus ditempatkan sebelum slot habis.
    """
    load = Counter(d for mk in courses for d in mk["dosen"])
    return sorted(
        courses,
        key=lambda mk: (len(mk["dosen"]), sum(load[d] for d in mk["dosen"])),
        reverse=True,
    )


def local_penalty(slot, room, dosen, used_rooms, used_dosen):
    """Penalti jika MK ditaruh di (slot, room): bentrok ruangan + bentrok dosen.

    Konsisten dengan calculate_fitness di main.py (1 poin per jenis bentrok
    per pasangan kelas).
    """
    penalty = 1 if (slot, room) in used_rooms else 0
    penalty += sum(1 for d in dosen if (slot, d) in used_dosen)
    return penalty


def greedy_schedule(courses):
    """Bangun jadwal: tiap MK dipilih pasangan (slot, ruangan) berpenalti terkecil.

    Pilihan bersifat lokal & final (tanpa backtracking). Tie-break: urutan
    slot lalu ruangan pada dataset, sehingga hasil deterministik.
    """
    used_rooms = set()   # {(slot, ruangan)}
    used_dosen = set()   # {(slot, dosen)}
    schedule = []

    for mk in order_courses(courses):
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

        _, slot, room = best
        used_rooms.add((slot, room))
        used_dosen.update((slot, d) for d in mk["dosen"])
        schedule.append({
            "id_mk": mk["id_mk"],
            "nama_mk": mk["nama_mk"],
            "dosen": mk["dosen"],
            "ruangan": room,
            "slot_waktu": slot,
        })
    return schedule


def calculate_penalty(schedule):
    """Hitung total bentrok (identik dengan metrik penalti di main.py)."""
    penalty = 0
    for i in range(len(schedule)):
        for j in range(i + 1, len(schedule)):
            a, b = schedule[i], schedule[j]
            if a["slot_waktu"] != b["slot_waktu"]:
                continue
            if a["ruangan"] == b["ruangan"]:
                penalty += 1
            if not set(a["dosen"]).isdisjoint(b["dosen"]):
                penalty += 1
    return penalty


# ==========================================
# 3. EKSEKUSI
# ==========================================
print("=" * 50)
print("MEMULAI PENJADWALAN DENGAN GREEDY ALGORITHM")
print("=" * 50)

start = time.perf_counter()
best_schedule = greedy_schedule(data_mk)
elapsed = time.perf_counter() - start

total_penalty = calculate_penalty(best_schedule)
fitness = 1.0 / (1.0 + total_penalty)

if total_penalty == 0:
    print(">>> JADWAL TANPA BENTROK DITEMUKAN <<<")
else:
    print(f"Greedy selesai dengan sisa bentrok: {total_penalty}")
print(f"Sisa Bentrok: {total_penalty} | Fitness: {fitness:.4f}")
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
