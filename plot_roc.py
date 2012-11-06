import numpy as np
import matplotlib.pyplot as plt
import os

import receiver_operating_characteristic as roc

from config import ConfigOpt
from opt import options

config_sources = options.config_source.split(',')
config_opts = [ConfigOpt(cs) for cs in config_sources]

feature_str = ''
for cos in config_opts:
    feature_str += cos.feature_str

if options.bootstrap:
    neg_file = os.path.join(config_opts[0].result_path,
                            "non_annotated_results_%s.%sb.npy" %
                                (options.set_type, feature_str))
    pos_file = os.path.join(config_opts[0].result_path,
                            "annotated_results_%s.%sb.npy" %
                                (options.set_type, feature_str))
else:
    neg_file = os.path.join(config_opts[0].result_path,
                            "non_annotated_results_%s.npy"%options.set_type)
    pos_file = os.path.join(config_opts[0].result_path,
                            "annotated_results_%s.npy"%options.set_type)

print("Read: {0}".format(pos_file))
pos_scores = np.load(pos_file)

print("Read: {0}".format(neg_file))
neg_scores = np.load(neg_file)

tp, fp, th = roc.compute_roc(pos_scores, neg_scores)

plt.plot(fp, 1-tp, '.', color='green', linewidth=2.0)
plt.ion()
plt.ylim(0.05, 0.3)
plt.xlim(2e-7, 1e-4)
plt.xscale('log', xbase=10)
plt.yscale('log', ybase=2)
plt.xlabel('False Positive Rate Per Window')
#yticks = np.logspace(np.log10(tp_min), np.log10(tp_max), 10)
#plt.yticks(yticks, ["%.2f" % y for y in yticks])
plt.yticks([.05, 0.1, 0.2, 0.3, 0.4, 0.5],
           [.05, 0.1, 0.2, 0.3, 0.4, 0.5])
plt.ylabel('Miss Rate')
plt.legend(fancybox=True,shadow=True, loc='lower left')
plt.grid()

plt.show()
