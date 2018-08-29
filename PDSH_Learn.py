import numpy as np
import pandas as pd
import random
import seaborn as sns
sns.set()
from datetime import datetime
from dateutil import parser
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

def f2(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

def showGraph():
    data = pd.read_csv('FremontBridge.csv', index_col='Date', parse_dates=True)
    data.columns = ['West', 'East']
    data['Total'] = data.eval('West + East')
    data.plot()

    daily = data.resample('D').sum()
    daily.rolling(50, center=True, win_type='gaussian').sum(std=10).plot(style=[':', '--', '-'])
    plt.ylabel('mean hourly count')

    #by_time = data.groupby(data.index.time).mean()
    #hourly_ticks = 4 * 60 * 60 * np.arange(6)
    #by_time.plot(xticks=hourly_ticks, style=[':', '--', '-'])

    by_weekday = data.groupby(data.index.dayofweek).mean()
    by_weekday.index = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
    by_weekday.plot(style=[':', '--', '-'])

    weekend = np.where(data.index.weekday < 5, 'Weekday', 'Weekend')
    by_time = data.groupby([weekend, data.index.time]).mean()
    hourly_ticks = 4 * 60 * 60 * np.arange(6)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    by_time.ix['Weekday'].plot(ax=ax[0], title='Weekdays',
                               xticks=hourly_ticks, style=[':', '--', '-'])
    by_time.ix['Weekend'].plot(ax=ax[1], title='Weekends',
                               xticks=hourly_ticks, style=[':', '--', '-'])

    plt.show()

def matplt_01():
    #plt.style.use('classic')
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    ax = plt.axes()
    x = np.linspace(0, 10, 1000)
    #plt.plot(x, np.sin(x - 0), color='blue')  # Задаем цвет по названию
    #plt.plot(x, np.sin(x - 1), color='g')  # Краткий код цвета (rgbcmyk)
    #plt.plot(x, np.sin(x - 2), color='0.75')  # Шкала оттенков серого цвета, значения в диапазоне от 0 до 1
    #plt.plot(x, np.sin(x - 3), color='#FFDD44')  # 16-ричный код (RRGGBB от 00 до FF)
    #plt.plot(x, np.sin(x - 4), color=(1.0, 0.2, 0.3))  # Кортеж RGB, значения 0 и 1
    #plt.plot(x, np.sin(x - 5), color='chartreuse')

    #plt.plot(x, x + 0, linestyle='solid')
    #plt.plot(x, x + 1, linestyle='dashed')
    #plt.plot(x, x + 2, linestyle='dashdot')
    #plt.plot(x, x + 3, linestyle='dotted')

    #plt.plot(x, np.sin(x))
    #plt.xlim(-1, 11)
    #plt.ylim(-1.5, 1.5)
    #plt.axis([-1, 11, -1.5, 1.5])
    #plt.title("A Sine Curve")  # Синусоидальная кривая
    #plt.xlabel("x")
    #plt.ylabel("sin(x)")

    plt.plot(x, np.sin(x), '-g', label='sin(x)')
    plt.plot(x, np.cos(x), ':b', label='cos(x)')
    plt.axis('equal')
    plt.legend()

    plt.show()

def matplt_02():
    plt.style.use('seaborn-whitegrid')
    x = np.linspace(0, 10, 30)
    y = np.sin(x)
    plt.plot(x, y, 'o', color='black')
    plt.axis('tight')
    plt.show()

def matplt_markers():
    rng = np.random.RandomState(0)
    for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
        plt.plot(rng.rand(5), rng.rand(5), marker,
                 label="marker='{0}'".format(marker))
    plt.legend(numpoints=1)
    plt.xlim(0, 1.8)
    plt.show()

def matplt_scatter():
    plt.style.use('seaborn-whitegrid')
    rng = np.random.RandomState(0)
    x = rng.randn(100)
    y = rng.randn(100)
    colors = rng.rand(100)
    sizes = 1000 * rng.rand(100)
    plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,
                cmap='viridis')
    plt.colorbar()
    plt.show()

def matplt_iris():
    from sklearn.datasets import load_iris
    iris = load_iris()
    features = iris.data.T
    plt.scatter(features[0], features[1], alpha=0.2,
                s=100 * features[3], c=iris.target, cmap='viridis')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.show()

def matplt_error():
    plt.style.use('seaborn-whitegrid')
    x = np.linspace(0, 10, 50)
    dy = 0.8
    y = np.sin(x) + dy * np.random.randn(50)
    #plt.errorbar(x, y, yerr=dy, fmt='.k')
    #plt.errorbar(x, y, yerr=dy, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0);
    plt.show()

def matplt_GPR():
    from sklearn.gaussian_process import GaussianProcess
    # Описываем модель и отрисовываем некоторые данные
    model = lambda x: x * np.sin(x)
    xdata = np.array([1, 3, 5, 6, 8])
    ydata = model(xdata)
    # Выполняем подгонку Гауссова процесса
    gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1E-1, random_start=100)
    gp.fit(xdata[:, np.newaxis], ydata)
    xfit = np.linspace(0, 10, 1000)
    yfit, MSE = gp.predict(xfit[:, np.newaxis], eval_MSE=True)
    dyfit = 2 * np.sqrt(MSE) # 2*сигма ~ область с уровнем доверия 95%
    # Визуализируем результат
    plt.plot(xdata, ydata, 'or')
    plt.plot(xfit, yfit, '-', color='gray')
    plt.fill_between(xfit, yfit - dyfit, yfit + dyfit, color='gray', alpha=0.2)
    plt.xlim(0, 10)
    plt.show()

def matplt_contour():
    plt.style.use('seaborn-white')
    # z = f(x, y)
    x = np.linspace(0, 5, 50)
    y = np.linspace(0, 5, 40)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    #plt.contour(X, Y, Z, colors='black')
    #plt.contour(X, Y, Z, 20, cmap='RdGy')
    plt.contourf(X, Y, Z, 20, cmap='RdGy')
    plt.colorbar()
    plt.show()

def matplt_imshow():
    plt.style.use('seaborn-white')
    # z = f(x, y)
    x = np.linspace(0, 5, 50)
    y = np.linspace(0, 5, 40)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    plt.axis(aspect='image')
    contours = plt.contour(X, Y, Z, 3, colors='black')
    plt.clabel(contours, inline=True, fontsize=8)
    plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower', cmap='RdGy', alpha=0.5)
    plt.colorbar()
    plt.show()

def matplt_hist_01():
    plt.style.use('seaborn-white')
    data = np.random.randn(1000)
    #plt.hist(data)
    plt.hist(data, bins=30, normed=True, alpha=0.5, histtype='stepfilled', color='steelblue', edgecolor='none')
    plt.show()

def matplt_hist_02():
    plt.style.use('seaborn-white')
    x1 = np.random.normal(0, 0.8, 1000)
    x2 = np.random.normal(-2, 1, 1000)
    x3 = np.random.normal(3, 2, 1000)
    kwargs = dict(histtype='stepfilled', alpha=0.3, normed=True, bins=40)
    plt.hist(x1, **kwargs)
    plt.hist(x2, **kwargs)
    plt.hist(x3, **kwargs)
    plt.show()

def count_hist():
    data = np.random.randn(1000)
    counts, bin_edges = np.histogram(data, bins=5)
    print(counts)

def matplt_hist2d():
    mean = [0, 0]
    cov = [[1, 1],[1, 2]]
    x, y = np.random.multivariate_normal(mean, cov, 10000).T
    plt.hist2d(x, y, bins=30, cmap = 'Blues')
    cb = plt.colorbar()
    cb.set_label('counts in bin') # Количество в интервале
    plt.show()

def matplt_hist_hex():
    mean = [0, 0]
    cov = [[1, 1],[1, 2]]
    x, y = np.random.multivariate_normal(mean, cov, 10000).T
    plt.hexbin(x, y, gridsize=30, cmap='Blues')
    cb = plt.colorbar(label='count in bin')  # Количество в интервале
    plt.show()

# kernel density estimation (KDE)
def matplt_kde():
    from scipy.stats import gaussian_kde
    mean = [0, 0]
    cov = [[1, 1], [1, 2]]
    x, y = np.random.multivariate_normal(mean, cov, 10000).T
    # Выполняем подбор на массиве размера [Ndim, Nsamples]
    data = np.vstack([x, y])
    kde = gaussian_kde(data)
    # Вычисляем на регулярной координатной сетке
    xgrid = np.linspace(-3.5, 3.5, 40)
    ygrid = np.linspace(-6, 6, 40)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
    # Выводим график результата в виде изображения
    plt.imshow(Z.reshape(Xgrid.shape), origin='lower', aspect='auto', extent=[-3.5, 3.5, -6, 6], cmap='Blues')
    cb = plt.colorbar()
    cb.set_label("density")  # Плотность
    plt.show()

def matplt_legend_01():
    import matplotlib.pyplot as plt
    plt.style.use('classic')
    x = np.linspace(0, 10, 1000)
    fig, ax = plt.subplots()
    ax.plot(x, np.sin(x), '-b', label='Sine')  # Синус
    ax.plot(x, np.cos(x), '--r', label='Cosine')  # Косинус
    ax.axis('equal')
    #ax.legend()
    #ax.legend(loc='upper left', frameon=False)
    #ax.legend(frameon=False, loc='lower center', ncol=2)
    ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.show()

def matplt_legend_02():
    import matplotlib.pyplot as plt
    plt.style.use('classic')
    x = np.linspace(0, 10, 1000)
    fig, ax = plt.subplots()
    y = np.sin(x[:, np.newaxis] + np.pi * np.arange(0, 2, 0.5))
    lines = plt.plot(x, y)
    # lines представляет собой список экземпляров класса plt.Line2D
    plt.legend(lines[:2], ['first', 'second'])
    plt.show()

def matplt_legend_03():
    cities = pd.read_csv('data/california_cities.csv')
    lat, lon = cities['latd'], cities['longd']
    population, area = cities['population_total'], cities['area_total_km2']
    # Распределяем точки по нужным местам,
    # с использованием размера и цвета, но без меток
    plt.scatter(lon, lat, label=None,
                c=np.log10(population), cmap='viridis',
                s=area, linewidth=0, alpha=0.5)
    plt.axis(aspect='equal')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.colorbar(label='log$_{10}$(population)')
    plt.clim(3, 7)
    # Создаем легенду:
    # выводим на график пустые списки с нужным размером и меткой
    for area in [100, 300, 500]:
        plt.scatter([], [], c='k', alpha=0.3, s=area, label=str(area) + ' km$^2$')
    plt.legend(scatterpoints=1, frameon=False,
               labelspacing=1, title='City Area')  # Города
    plt.title('California Cities: Area and Population') # Города Калифорнии: местоположение и население
    plt.show()

def matplt_multilegend():
    fig, ax = plt.subplots()
    lines = []
    styles = ['-', '--', '-.', ':']
    x = np.linspace(0, 10, 1000)
    for i in range(4):
        lines += ax.plot(x, np.sin(x - i * np.pi / 2), styles[i], color='black')
    ax.axis('equal')
    # Задаем линии и метки первой легенды
    ax.legend(lines[:2], ['line A', 'line B'],  # Линия А, линия B
              loc='upper right', frameon=False)
    # Создаем вторую легенду и добавляем рисователь вручную
    from matplotlib.legend import Legend
    leg = Legend(ax, lines[2:], ['line C', 'line D'],  # Линия С, линия D
                 loc='lower right', frameon=False)
    ax.add_artist(leg)
    plt.show()

def matplt_color():
    plt.style.use('classic')
    x = np.linspace(0, 10, 1000)
    I = np.sin(x) * np.cos(x[:, np.newaxis])
    #plt.imshow(I, cmap='gray')
    plt.imshow(I, cmap=plt.cm.get_cmap('Blues', 6))
    plt.colorbar()
    plt.clim(-1, 1)
    plt.show()

def matplt_subplots():
    plt.style.use('seaborn-white')
    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
                       xticklabels=[], ylim=(-1.2, 1.2))
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],
                       ylim=(-1.2, 1.2))
    x = np.linspace(0, 10)
    ax1.plot(np.sin(x))
    ax2.plot(np.cos(x))
    plt.show()

def ex_biths():
    births = pd.read_csv('births.csv')
    quartiles = np.percentile(births['births'], [25, 50, 75])
    mu, sig = quartiles[1], 0.74 * (quartiles[2] - quartiles[0])
    births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')
    births['day'] = births['day'].astype(int)
    births.index = pd.to_datetime(10000 * births.year +
                                  100 * births.month +
                                  births.day, format='%Y%m%d')
    births_by_date = births.pivot_table('births',
                                        [births.index.month, births.index.day])
    births_by_date.index = [pd.datetime(2012, month, day)
                            for (month, day) in births_by_date.index]
    fig, ax = plt.subplots(figsize=(12, 4))
    births_by_date.plot(ax=ax)
    fig, ax = plt.subplots(figsize=(12, 4))
    births_by_date.plot(ax=ax)
    # Добавляем метки на график
    style = dict(size=10, color='gray')
    ax.text('2012-1-1', 3950, "New Year's Day", **style)
    ax.text('2012-7-4', 4250, "Independence Day", ha='center', **style)
    ax.text('2012-9-4', 4850, "Labor Day", ha='center', **style)
    ax.text('2012-10-31', 4600, "Halloween", ha='right', **style)
    ax.text('2012-11-25', 4450, "Thanksgiving", ha='center', **style)
    ax.text('2012-12-25', 3850, "Christmas ", ha='right', **style)
    # Добавляем метки для осей координат
    ax.set(title='USA births by day of year (1969-1988)',
           ylabel='average daily births')
    # Размечаем ось X центрированными метками для месяцев
    ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
    ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'))
    plt.show()

def matplt_arrow():
    fig, ax = plt.subplots()
    x = np.linspace(0, 20, 1000)
    ax.plot(x, np.cos(x))
    ax.axis('equal')
    ax.annotate('local maximum', xy=(6.28, 1), xytext=(10, 4),
                arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate('local minimum', xy=(5 * np.pi, -1), xytext=(2, -6),
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle3,angleA=0,angleB=-90"))
    plt.show()

def matplt_locator():
    ax = plt.axes()
    ax.plot(np.random.rand(50))
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    plt.show()

def format_func(value, tick_number):
    # Определяем количество кратных пи/2 значений
    # [в требуемом диапазоне]
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)

def matplt_locator_sin():
    # Строим графики синуса и косинуса
    fig, ax = plt.subplots()
    x = np.linspace(0, 3 * np.pi, 1000)
    ax.plot(x, np.sin(x), lw=3, label='Sine')
    ax.plot(x, np.cos(x), lw=3, label='Cosine')
    # Настраиваем сетку, легенду и задаем пределы осей координат
    ax.grid(True)
    ax.legend(frameon=False)
    ax.axis('equal')
    ax.set_xlim(0, 3 * np.pi)
    #ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
    plt.show()

def hist_and_lines():
    plt.style.use('seaborn-pastel')
    np.random.seed(0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(np.random.randn(1000))
    for i in range(3):
        ax[1].plot(np.random.rand(10))
        ax[1].legend(['a', 'b', 'c'], loc='lower left')
    plt.show()

def matplt_3d_pure():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # Данные для трехмерной кривой
    zline = np.linspace(0, 15, 1000)
    xline = np.sin(zline)
    yline = np.cos(zline)
    ax.plot3D(xline, yline, zline, 'gray')
    # Данные для трехмерных точек
    zdata = 15 * np.random.random(100)
    xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
    ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
    plt.show()

def matplt_3d_contour():
    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)
    X, Y = np.meshgrid(x, y)
    Z = f2(X, Y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(60, 35)
    plt.show()

def matplt_3d_wireframe():
    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)
    X, Y = np.meshgrid(x, y)
    Z = f2(X, Y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(X, Y, Z, color='black')
    ax.set_title('wireframe')  # Каркас
    plt.show()

def matplt_3d_surface():
    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)
    X, Y = np.meshgrid(x, y)
    Z = f2(X, Y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    cmap='viridis', edgecolor='none')
    ax.set_title('surface') # Поверхность
    plt.show()


def matplt_3d_surface_p():
    r = np.linspace(0, 6, 20)
    theta = np.linspace(-0.9 * np.pi, 0.8 * np.pi, 40)
    r, theta = np.meshgrid(r, theta)
    X = r * np.sin(theta)
    Y = r * np.cos(theta)
    Z = f(X, Y)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    plt.show()

def matplt_3d_triangulation():
    theta = 2 * np.pi * np.random.random(1000)
    r = 6 * np.random.random(1000)
    x = np.ravel(r * np.sin(theta))
    y = np.ravel(r * np.cos(theta))
    z = f2(x, y)
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
    plt.show()

def matplt_3d_moebius():
    theta = np.linspace(0, 2 * np.pi, 30)
    w = np.linspace(-0.25, 0.25, 8)
    w, theta = np.meshgrid(w, theta)
    phi = 0.5 * theta
    # радиус в плоскости X-Y
    r = 1 + w * np.cos(phi)
    x = np.ravel(r * np.cos(theta))
    y = np.ravel(r * np.sin(theta))
    z = np.ravel(w * np.sin(phi))
    # Выполняем триангуляцию в координатах базовой параметризации
    from matplotlib.tri import Triangulation
    tri = Triangulation(np.ravel(w), np.ravel(theta))
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x, y, z, triangles=tri.triangles,
                    cmap='viridis', linewidths=0.2)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.show()

def seaborn_simple():
    rng = np.random.RandomState(0)
    x = np.linspace(0, 10, 500)
    y = np.cumsum(rng.randn(500, 6), 0)
    plt.plot(x, y)
    plt.legend('ABCDEF', ncol=2, loc='upper left')
    plt.show()

def matplt_hist_simple():
    data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]],
                                         size=2000)
    data = pd.DataFrame(data, columns=['x', 'y'])
    #for col in 'xy':
    #    plt.hist(data[col], normed=True, alpha=0.5)
    #for col in 'xy':
    #    sns.kdeplot(data[col], shade=True)
    #sns.distplot(data['x'])
    #sns.distplot(data['y'])
    #with sns.axes_style('white'):
    #    sns.jointplot("x", "y", data, kind='kde')
    with sns.axes_style('white'):
        sns.jointplot("x", "y", data, kind='hex')
    plt.show()

def seaborn_iris():
    iris = sns.load_dataset("iris")
    print(iris.head())
    sns.pairplot(iris, hue='species', size=2.5)
    plt.show()

def seaborn_tips():
    tips = sns.load_dataset('tips')
    print(tips.head())
    tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill']
    grid = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True)
    grid.map(plt.hist, "tip_pct", bins=np.linspace(0, 40, 15))
    with sns.axes_style(style='ticks'):
        g = sns.factorplot("day", "total_bill", "sex", data=tips,
                           kind="box")
    g.set_axis_labels("Day", "Total Bill")  # День; Итого
    with sns.axes_style('white'):
        sns.jointplot("total_bill", "tip", data=tips, kind='hex')
    sns.jointplot("total_bill", "tip", data=tips, kind='reg')
    plt.show()

def seaborn_planets():
    planets = sns.load_dataset('planets')
    print(planets.head())
    with sns.axes_style('white'):
        g = sns.factorplot("year", data=planets, aspect=2,  # Год
                           kind="count", color='steelblue')  # Количество
    g.set_xticklabels(step=5)
    with sns.axes_style('white'):
        g = sns.factorplot("year", data=planets, aspect=4.0, kind='count',
                           hue='method', order=range(2001, 2015))
    g.set_ylabels('Number of Planets Discovered') # Количество обнаруженных планет
    plt.show()

# !curl -O https://raw.githubusercontent.com/jakevdp/marathon-data/
# master/marathon-data.csv
def seaborn_marathon():
    print('Test')

def main():
    #matplt_markers()
    #matplt_02()
    #print(data.dropna().describe())
    #matplt_scatter()
    #matplt_iris()
    #matplt_error()
    #matplt_GPR()
    #matplt_contour()
    #matplt_imshow()
    #matplt_hist_01()
    #matplt_hist_02()
    #count_hist()
    #matplt_hist2d()
    #matplt_hist_hex()
    #matplt_kde()
    #matplt_legend_01()
    #matplt_legend_02()
    #matplt_legend_03()
    #matplt_multilegend()
    #matplt_color()
    #matplt_subplots()
    #ex_biths()
    #matplt_arrow()
    #matplt_locator()
    #matplt_locator_sin()
    #hist_and_lines()
    #matplt_3d_pure()
    #matplt_3d_contour()
    #matplt_3d_wireframe()
    #matplt_3d_surface()
    #matplt_3d_surface_p()
    #matplt_3d_triangulation()
    #seaborn_simple()
    #matplt_hist_simple()
    #seaborn_iris()
    #seaborn_tips()
    seaborn_planets()

if __name__ == '__main__':
    main()