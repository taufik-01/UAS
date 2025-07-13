import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Baca file CSV yang sudah kamu update
data = pd.read_csv('student_marks_updated.csv')

# Tentukan variabel X dan Y
x = data['Test_1']
y = data['Test_2']

# Scatter plot dengan garis regresi
sns.regplot(x=x, y=y)
plt.title('Regresi Linier Test_1 vs Test_2')
plt.xlabel('Test_1')
plt.ylabel('Test_2')
plt.show()

# Hitung regresi linier manual
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print('Slope:', slope)
print('Intercept:', intercept)
print('R-squared:', r_value**2)