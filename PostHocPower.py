import statsmodels.stats.power as smp


def calculate_unpaired_ttest_php(es, n_per_group):

    power = smp.TTestIndPower().power(effect_size=es, nobs1=n_per_group, alpha=0.05,
                                      df=2*n_per_group-2, ratio=1, alternative='two-sided')

    print(power)


def calculate_paired_ttest_php(es, n_per_group):

    power = smp.TTestPower().power(effect_size=es, alpha=0.05, nobs=n_per_group,
                                   df=2*n_per_group-1, alternative='two-sided')

    print("Power = {}".format(round(power, 3)))


calculate_paired_ttest_php(es=.386, n_per_group=10)

