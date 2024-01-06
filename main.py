import os
import cv2
import numpy as np
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk

class PengenalanWajah:
    def __init__(self, master):
        self.master = master
        master.title("Pengenalan Wajah")

        self.folder_dataset = None
        self.path_gambar_input = None

        self.tombol_muat_data = Button(master, text="Muat Dataset", command=self.muat_dataset)
        self.tombol_muat_data.pack()

        self.tombol_muat_gambar = Button(master, text="Muat Gambar", command=self.muat_gambar)
        self.tombol_muat_gambar.pack()

        self.tombol_pengecekan = Button(master, text="Cek Wajah", command=self.cek_wajah)
        self.tombol_pengecekan.pack()

        self.label_hasil = Label(master, text="")
        self.label_hasil.pack()

        self.label_gambar = Label(master)
        self.label_gambar.pack()

    def muat_dataset(self):
        self.folder_dataset = filedialog.askdirectory()
        print("Dataset dimuat dari:", self.folder_dataset)

    def muat_gambar(self):
        self.path_gambar_input = filedialog.askopenfilename(filetypes=[("File gambar", "*.jpg;*.png")])
        print("Gambar dimuat dari:", self.path_gambar_input)
        self.tampil_gambar()

    def cek_wajah(self):
        if self.folder_dataset is None or self.path_gambar_input is None:
            print("Mohon muat dataset dan gambar terlebih dahulu.")
            return

        hasil = self.pengecekan_wajah()

        # Tampilkan hasil
        self.tampilkan_hasil(hasil)

    def pengecekan_wajah(self):
        recognizer = cv2.face.EigenFaceRecognizer_create()

        # Mempersiapkan dataset
        images, labels = self.persiapkan_dataset()

        if len(set(labels)) < 2:
            print("Memerlukan minimal dua label.")
            return {"label": None, "jarak": None}

        # Melatih model recognizer
        recognizer.train(images, np.array(labels))

        # Membaca gambar input
        gambar_input = cv2.imread(self.path_gambar_input, cv2.IMREAD_GRAYSCALE)
        gambar_input = cv2.resize(gambar_input, (100, 100))

        # Menghitung eigenface dan jarak Euclidean
        label_terdekat, jarak = recognizer.predict(gambar_input)

        return {"label": label_terdekat, "jarak": jarak}

    def persiapkan_dataset(self):
        images = []
        labels = []
        label_mapping = {}  # Kamus untuk memetakan nama folder ke nilai numerik

        label_counter = 0
        for root, dirs, files in os.walk(self.folder_dataset):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    path_gambar = os.path.join(root, file)
                    label_folder = os.path.basename(root)  # Gunakan nama folder sebagai label

                    # Jika label belum ada dalam label_mapping, tambahkan
                    if label_folder not in label_mapping:
                        label_mapping[label_folder] = label_counter
                        label_counter += 1

                    label_gambar = label_mapping[label_folder]

                    gambar = cv2.imread(path_gambar, cv2.IMREAD_GRAYSCALE)
                    gambar = cv2.resize(gambar, (100, 100))

                    images.append(gambar)
                    labels.append(label_gambar)
        return images, labels

    def tampil_gambar(self):
        gambar = Image.open(self.path_gambar_input)
        gambar = gambar.resize((300, 300), Image.ANTIALIAS)
        gambar = ImageTk.PhotoImage(gambar)
        self.label_gambar.config(image=gambar)
        self.label_gambar.image = gambar

    def tampilkan_hasil(self, hasil):
        if hasil["label"] is not None:
            pesan = f"Hasil: {hasil['label']}, Jarak: {hasil['jarak']:.2f}"
        else:
            pesan = "Tidak ada hasil yang sesuai."

        # Update label_hasil
        self.label_hasil.config(text=pesan)

def utama():
    root = Tk()
    app = PengenalanWajah(root)
    root.mainloop()

if __name__ == "__main__":
    utama()
