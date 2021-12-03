# количество стран, где показатель смертности
# (отношение числа всех умерших к числу всех заболевших) выше заданного значения

import pandas as pd
from multiprocessing import Pool
import time
from matplotlib import pyplot as plt

my_df = pd.read_csv("data.csv", sep=',', encoding="cp1251")

#print(df.info)
print(my_df.columns)
# print(my_df.loc[3])


'''df['Заболели'] = df['Заболели'].apply(lambda _str: int(_str.replace("'", "").strip()))
df = df[df['Активные случаи'] >= 0]

parameter_value = df.loc[2]
print(parameter_value, end="\n\n")
z = df[df["Умерли"]/df["Заболели"] > parameter_value["Умерли"]/parameter_value["Заболели"]]
print(z) '''


def try_with_multiproc(num_of_proc):
    start_time = time.time()
    with Pool(num_of_proc) as p:
        #global df
        df = my_df.copy()
        df['Заболели'] = df['Заболели'].apply(lambda _str: int(_str.replace("'", "").strip()))
        df = df[df['Активные случаи'] >= 0]

        parameter_value = df.loc[2]
        df = df[df["Умерли"] / df["Заболели"] > parameter_value["Умерли"] / parameter_value["Заболели"]]

    end_time = time.time()
    #print(end_time - start_time)
    return end_time - start_time


a = try_with_multiproc(1)
b = try_with_multiproc(2)
c = try_with_multiproc(3)
d = try_with_multiproc(4)
e = try_with_multiproc(5)
f = try_with_multiproc(6)
g = try_with_multiproc(7)
h = try_with_multiproc(8)
print(f'time difference (1 processes - 2 process): {a - b}')

print(f'calculation made by 1 process: {a}')
print(f'calculation made by 2 processes: {b}')
print(f'calculation made by 3 processes: {c}')
print(f'calculation made by 4 processes: {d}')
print(f'calculation made by 5 processes: {e}')
print(f'calculation made by 6 processes: {f}')
print(f'calculation made by 7 processes: {g}')
print(f'calculation made by 8 processes: {h}')

'''step = .1
x = np.arange(-10,10.1,step)
y1 = np.sinc(x)

#линейный график
fig=plt.figure(figsize=(30, 5))
plt.ylabel('y')
plt.xlabel('x')
plt.title('y=sinc(x)')
plt.plot(x,y1, color='green', linestyle = '--', marker='x', linewidth=1, markersize=4 )
plt.show()'''

plt.ylabel('y')
plt.xlabel('x')
plt.title('time by number of processes')
plt.plot([0, a, b, c, d, e, f, g, h], color='green')
plt.show()