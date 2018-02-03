# Python
Python Scripts
# Calculation of power lines
# Alexander Litvintsev
# alexanderlitvintsev@mail.ru

import math
import numpy as np
from numpy.lib.scimath import logn

# variables

# Удельная проводимость однородной земли, См/м
init_var_y = 0.01

# Магнитная постоянная
init_var_u = 0.00000125663706

# Частота тока, Гц
init_var_f = 50

# Круговая частота, 1/с
init_var_w = 2 * math.pi * init_var_f

# Длина линии, км
init_var_l = 50

# Сопротивление одного км провода постоянному току, Ом
init_var_r0 = 0.122

# Площадь сечения провода, мм2
init_var_s = 240

# Эквивалентный радиус провода, м
init_var_r = 0.012

# Высота подвеса провода,м
init_var_h = 19

# Длина пролета, м
init_var_l_a = 180

# Линейный вес, даН/м
init_var_p_weight = 0.921

# Механическое тяжение, даН
init_var_t = 1872

# Стрела провеса, м
var_fm = (math.pow(init_var_l_a, 2) * init_var_p_weight) / (8 * init_var_t)

# Высота провода над землей, м
var_h1 = init_var_h - (2/3) * var_fm
var_h2 = init_var_h - (2/3) * var_fm
var_h3 = init_var_h - (2/3) * var_fm

# Координаты расположения проводов с учетом стрелы провеса
var_x1 = -5
var_x2 = 0
var_x3 = 5
var_y1 = 19
var_y2 = 19
var_y3 = 19

# Расстояние между проводами
var_d12 = math.sqrt(math.pow(var_x1-var_x2, 2) + math.pow(var_y1-var_y2, 2))
var_d13 = math.sqrt(math.pow(var_x1-var_x3, 2) + math.pow(var_y1-var_y3, 2))
var_d23 = math.sqrt(math.pow(var_x2-var_x3, 2) + math.pow(var_y2-var_y3, 2))

var_D12 = math.sqrt(math.pow(var_x1-var_x2, 2) + math.pow(var_y1+var_y2, 2))
var_D13 = math.sqrt(math.pow(var_x1-var_x3, 2) + math.pow(var_y1+var_y3, 2))
var_D23 = math.sqrt(math.pow(var_x2-var_x3, 2) + math.pow(var_y2+var_y3, 2))

# Расчет сопротивлений
Z_out_11 = (init_var_w * init_var_u / 8) + (((init_var_w * init_var_u) / (2 * math.pi)) * logn( math.e, 1.85 / (init_var_r * math.sqrt(init_var_y * init_var_w * init_var_u)) ))*1j
Z_inn_11 = init_var_r0 * (0.9 + 0.0063 * math.pow(init_var_f , 0.755)) + (0.001 * ((0.033 - 0.00107 * math.pow(init_var_f, 0.83)) * init_var_s + (1.07 * math.pow(init_var_f, 0.83) - 13.5)) )*1j
Z_11 = Z_out_11 * 1000 + Z_inn_11

Z_12 = (init_var_w * init_var_u / 8) + (((init_var_w * init_var_u) / (2 * math.pi)) * logn( math.e, 1.85 / (var_d12 * math.sqrt(init_var_y * init_var_w * init_var_u)) ))*1j
Z_12 *= 1000

Z_13 = (init_var_w * init_var_u / 8) + (((init_var_w * init_var_u) / (2 * math.pi)) * logn( math.e, 1.85 / (var_d13 * math.sqrt(init_var_y * init_var_w * init_var_u)) ))*1j
Z_13 *= 1000

Z_21 = (init_var_w * init_var_u / 8) + (((init_var_w * init_var_u) / (2 * math.pi)) * logn( math.e, 1.85 / (var_d12 * math.sqrt(init_var_y * init_var_w * init_var_u)) ))*1j
Z_21 *= 1000

Z_out_22 = (init_var_w * init_var_u / 8) + (((init_var_w * init_var_u) / (2 * math.pi)) * logn( math.e, 1.85 / (init_var_r * math.sqrt(init_var_y * init_var_w * init_var_u)) ))*1j
Z_inn_22 = init_var_r0 * (0.9 + 0.0063 * math.pow(init_var_f , 0.755)) + (0.001 * ((0.033 - 0.00107 * math.pow(init_var_f, 0.83)) * init_var_s + (1.07 * math.pow(init_var_f, 0.83) - 13.5)) )*1j
Z_22 = Z_out_22 * 1000 + Z_inn_22

Z_23 = (init_var_w * init_var_u / 8) + (((init_var_w * init_var_u) / (2 * math.pi)) * logn( math.e, 1.85 / (var_d23 * math.sqrt(init_var_y * init_var_w * init_var_u)) ))*1j
Z_23 *= 1000

Z_31 = (init_var_w * init_var_u / 8) + (((init_var_w * init_var_u) / (2 * math.pi)) * logn( math.e, 1.85 / (var_d13 * math.sqrt(init_var_y * init_var_w * init_var_u)) ))*1j
Z_31 *= 1000

Z_32 = (init_var_w * init_var_u / 8) + (((init_var_w * init_var_u) / (2 * math.pi)) * logn( math.e, 1.85 / (var_d23 * math.sqrt(init_var_y * init_var_w * init_var_u)) ))*1j
Z_32 *= 1000

Z_out_33 = (init_var_w * init_var_u / 8) + (((init_var_w * init_var_u) / (2 * math.pi)) * logn( math.e, 1.85 / (init_var_r * math.sqrt(init_var_y * init_var_w * init_var_u)) ))*1j
Z_inn_33 = init_var_r0 * (0.9 + 0.0063 * math.pow(init_var_f , 0.755)) + (0.001 * ((0.033 - 0.00107 * math.pow(init_var_f, 0.83)) * init_var_s + (1.07 * math.pow(init_var_f, 0.83) - 13.5)) )*1j
Z_33 = Z_out_33 * 1000 + Z_inn_33

Z = np.matrix([[Z_11, Z_12, Z_13], [Z_21, Z_22, Z_23], [Z_31, Z_32, Z_33]])
Z = Z * init_var_l

D = np.linalg.inv(Z)
Yrc1 = np.hstack((D, -D))
Yrc2 = np.hstack((-D, D))
Yrc = np.vstack((Yrc1, Yrc2))

# Эквивалентный радиус провода, см
var_r = init_var_r * 100

a11 = 1.8 * pow(10, 7) * logn(math.e, (200 * var_h1) / var_r)
a12 = 1.8 * pow(10, 7) * logn(math.e, var_D12 / var_d12)
a13 = 1.8 * pow(10, 7) * logn(math.e, var_D13 / var_d13)
a21 = a12
a22 = 1.8 * pow(10, 7) * logn(math.e, (200 * var_h2) / var_r)
a23 = 1.8 * pow(10, 7) * logn(math.e, var_D23 / var_d23)
a31 = a13
a32 = a23
a33 = 1.8 * pow(10, 7) * logn(math.e, (200 * var_h3) / var_r)

a = np.matrix([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
b = np.linalg.inv(a)

m_zero = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

Cy1 = np.hstack((b, m_zero))
Cy2 = np.hstack((m_zero, b))
Cy = np.vstack((Cy1, Cy2))

Cy = 0.5 * init_var_l * Cy

Yc = Yrc + init_var_w * Cy * 1j

Yc_L = Yc

Yc_L_h1, Yc_L_h2 = np.hsplit(Yc_L, 2)
Yc_L_v1, Yc_L_v2 = np.vsplit(Yc_L_h2, 2)

Y12 = Yc_L_v1
Y22 = Yc_L_v2

Uf = 110000 / math.sqrt(3)

U1 = Uf * math.cos((0/180) * math.pi) + Uf * math.sin((0/180) * math.pi) * 1j
U2 = Uf * math.cos((-120/180) * math.pi) + Uf * math.sin((-120/180) * math.pi) * 1j
U3 = Uf * math.cos((120/180) * math.pi) + Uf * math.sin((120/180) * math.pi) * 1j
U4 = Uf * math.cos((0/180) * math.pi) + Uf * math.sin((0/180) * math.pi) * 1j
U5 = Uf * math.cos((-120/180) * math.pi) + Uf * math.sin((-120/180) * math.pi) * 1j
U6 = Uf * math.cos((120/180) * math.pi) + Uf * math.sin((120/180) * math.pi) * 1j

S = 20000000 + 20000000*1j

U123 = np.matrix([[U1], [U2], [U3]])
U456 = np.matrix([[U4], [U5], [U6]])

for i in range(1,20):
    I456 = - (S / U456).real + (S / U456).imag * 1j
    U = np.linalg.inv(Y22) * (I456 - Y12 * U123)
    U456 = U

print('Фазные напряжения на конце ЛЭП: ', U)




