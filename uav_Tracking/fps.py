import matplotlib.pyplot as plt


labels = ["MOSSE", "CSRT",  "KCF", "MEDIAN", "TLD","GOTURN","BOOSTING", "MIL"]
fps_values = [average_fps_mosse, average_fps_csrt,  average_fps_kcf,
              average_fps_medianflow, average_fps_tld,average_fps_goturn, average_fps_boosting, average_fps_mil]

plt.bar(labels, fps_values, color=['green', 'red', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'royalblue'])

plt.ylabel("Ortalama FPS Değeri")
plt.title("Algoritmaların Ortalama FPS Değerleri")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)  # X eksenindeki etiketleri 45 derece döndürme
plt.show()
