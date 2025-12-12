import numpy as np
def R_correlation(y1, y2):
    m_y1 = np.mean(y1)
    m_y2 = np.mean(y2)

    a = np.sum((y2 - m_y2) * (y1 - m_y1))
    b = np.sqrt(np.sum((y1 - m_y1) ** 2)) * np.sqrt(np.sum((y2 - m_y2) ** 2))
    r2 = (a / (b+1.0e-6)) ** 2

    rsme = np.sqrt(np.mean((y1 - y2) ** 2))
    return r2, rsme
