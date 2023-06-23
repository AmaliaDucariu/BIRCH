import matplotlib.pyplot as plt

cluster_no = [3, 4, 5, 6]
time_jobs = [2.050669, 2.083345, 2.038026, 2.058638]
time_food = [0.102816, 0.100737, 0.100021, 0.101329]

plt.plot(cluster_no, time_jobs, color='g', label='Joburi')
plt.plot(cluster_no, time_food, color='c', label='Mâncare')

plt.xlabel("Număr de clustere")
plt.ylabel("Timpi de Execuție (s)")

plt.legend()
plt.show()