import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv('student_marks.csv')

# Ambil kolom yang diperlukan
x = data['Test_1']
y = data['Test_2']

# Tampilkan statistik deskriptif
print('\n===== Statistik Deskriptif =====')
print('Test_1:')
print('Mean:', x.mean())
print('Median:', x.median())
print('Standard Deviation:', x.std())

print('\nTest_2:')
print('Mean:', y.mean())
print('Median:', y.median())
print('Standard Deviation:', y.std())

# Reshape x untuk regresi
x_reshaped = x.values.reshape(-1,1)

# Regresi Linier
model = LinearRegression()
model.fit(x_reshaped, y)

# Prediksi
y_pred = model.predict(x_reshaped)

# Hasil regresi
print('\n===== Hasil Regresi Linier =====')
print('Slope:', model.coef_[0])
print('Intercept:', model.intercept_)
print('R-squared:', r2_score(y, y_pred))

# Visualisasi
sns.lmplot(x='Test_1', y='Test_2', data=data)
plt.title('Regresi Linier Test_1 vs Test_2')
plt.show()