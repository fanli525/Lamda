import statsmodels.api as sm
import numpy as np
import scipy.stats as sps
from ConfigSpace import UniformFloatHyperparameter,  UniformIntegerHyperparameter,ConfigurationSpace

def get_hyperparameter_search_space():
    cs = ConfigurationSpace()

    dropout_1 = UniformFloatHyperparameter(
        "dropout_1", 0.0, 0.6, log=False, default_value=0.5)
    dropout_2 = UniformFloatHyperparameter(
        "dropout_2", 0.0, 0.6, log=False, default_value=0.5)
    cs.add_hyperparameters(
        [dropout_1, dropout_2])
    return cs
class FitPDF:
    def __init__(self, configspace,train_data_good, kde_vartypes, min_bandwidth=.3, bandwidth_factor=3,   bw_estimation = 'normal_reference'):
        self.train_data_good = train_data_good
        self.kde_vartypes = kde_vartypes
        self.min_bandwidth = min_bandwidth
        self.kde_models = {}
        self.bw_estimation=bw_estimation
        self.configspace = configspace
        self.bw_factor = bandwidth_factor

        hps = self.configspace.get_hyperparameters()

        self.kde_vartypes = ""
        self.vartypes = []

        for h in hps:
            if hasattr(h, 'sequence'):
                raise RuntimeError(
                    'This version on BOHB does not support ordinal hyperparameters. Please encode %s as an integer parameter!' % (
                        h.name))

            if hasattr(h, 'choices'):
                self.kde_vartypes += 'u'
                self.vartypes += [len(h.choices)]
            else:
                self.kde_vartypes += 'c'
                self.vartypes += [0]

        self.vartypes = np.array(self.vartypes, dtype=int)
    def fit_pdf_pro(self):
        good_pro = sm.nonparametric.KDEMultivariate(data=self.train_data_good, var_type=self.kde_vartypes, bw=self.bw_estimation)

        good_pro.bw = np.clip(good_pro.bw, self.min_bandwidth, None)

        self.kde_models = {
            'pro': good_pro,
        }

    def sample_from_pro(self, n_samples=100):
        self.num_samples = n_samples
        kde_good = self.kde_models['pro']
        vector_all=[]
        for i in range(self.num_samples):
            idx = np.random.randint(0, len(kde_good.data))
            datum = kde_good.data[idx]
            vector = []

            for m, bw, t in zip(datum, kde_good.bw, self.vartypes):
                bw = max(bw, self.min_bandwidth)
                if t == 0:
                    bw = self.bw_factor * bw
                    try:
                        vector.append(sps.truncnorm.rvs(-m / bw, (1 - m) / bw, loc=m, scale=bw))
                    except:
                        self.logger.warning(
                            "Truncated Normal failed for:\ndatum=%s\nbandwidth=%s\nfor entry with value %s" % (
                            datum, kde_good.bw, m))
                        self.logger.warning("data in the KDE:\n%s" % kde_good.data)
                else:
                    if np.random.rand() < (1 - bw):
                        vector.append(int(m))
                    else:
                        vector.append(np.random.randint(t))
            vector_all.append(vector)
        # Use KDEMultivariate's resample method to draw samples
        return vector_all
    def sample_weighted(self, data_min,data_max,n_samples=100, alpha=0.5, n_attempts=10000):
        """
        从加权后的分布中采样，结合 KDE 和均匀分布。

        Args:
            n_samples (int): 需要采样的样本数。
            alpha (float): KDE 的权重 (0 <= alpha <= 1)。
            n_attempts (int): 每次生成的候选样本数量。

        Returns:
            np.ndarray: 从加权分布中采样的样本。
        """
        if 'pro' not in self.kde_models:
            raise ValueError("The KDE model has not been fitted. Please call `fit_pdf_pro` first.")

        kde_good = self.kde_models['pro']
        # data_min = self.train_data_good.min(axis=0)
        # data_max = self.train_data_good.max(axis=0)

        samples = []
        while len(samples) < n_samples:
            # Step 1: Generate candidate points uniformly
            candidates = np.random.uniform(data_min, data_max, size=(n_attempts, self.train_data_good.shape[1]))

            # Step 2: Compute KDE and uniform PDFs
            kde_pdf = kde_good.pdf(candidates.T)
            uniform_pdf = np.ones(n_attempts) / np.prod(data_max - data_min)

            # Step 3: Weighted PDF
            weighted_pdf = alpha * kde_pdf + (1 - alpha) * uniform_pdf

            # Step 4: Rejection sampling
            acceptance_probs = weighted_pdf / weighted_pdf.max()
            accepted = candidates[np.random.rand(n_attempts) < acceptance_probs]

            # Collect enough samples
            samples.extend(accepted[:n_samples - len(samples)])

        return np.array(samples[:n_samples])
    # 示例数据


train_data_good = np.random.rand(100, 2)  # 2D 数据
kde_vartypes = 'cc'  # 连续变量类型
cs=get_hyperparameter_search_space()
# 初始化并拟合分布
pdf_fitter = FitPDF(cs,train_data_good, kde_vartypes)
pdf_fitter.fit_pdf_pro()

# 从加权分布中采样
weighted_samples = pdf_fitter.sample_weighted(data_min=np.array([0,0]),data_max=np.array([1,1]),n_samples=10, alpha=0.7)







