def loglogplot(hash_matrix, filename):
    from collections import Counter
    import matplotlib.pyplot as plt
    hm = hash_matrix
    cnt = Counter(hm)
    V = sorted(cnt.values(), reverse=True)
    V = np.array(V)
    fig = plt.figure(figsize=(15, 15))
    plt.bar(np.arange(len(V)), V)
    plt.ylabel("frequency")
    plt.xlabel("fixed point")
    plt.legend()
    plt.xlim((0, len(cnt)))
    plt.title("Frequency of fixed points")
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(filename)
    plt.close(fig)

def powerlawplot(hash_matrix, folder, force=False):
    from collections import Counter
    import powerlaw
    import matplotlib.pyplot as plt

    cat = {}
    filenamenoxmin = os.path.join(folder, "powerlawnoxmin.png")
    cat["powerlawnoxmin"] = filenamenoxmin

    filenamexminfull = os.path.join(folder, "powerlawxmin_full.png")
    cat["powerlawxminfull"] = filenamexminfull

    filenamexminwindow = os.path.join(folder, "powerlawxmin_window.png")
    cat["powerlawxminwindow"] = filenamexminwindow

    hm = hash_matrix
    cnt = Counter(hm)
    s = np.argsort(cnt.values())[::-1]
    K = cnt.keys()
    K = [K[s[i]] for i in range(len(K))]
    K_to_int = {k: i + 1 for i, k in enumerate(K)}
    x = [K_to_int[v] for v in hm]
    x = np.array(x)
    if not os.path.exists(filenamenoxmin) or force:
        fit = powerlaw.Fit(x, discrete=True, xmin=x.min())
        plt.clf()
        fig = plt.figure()
        try:
            fig2 = fit.plot_pdf(original_data=True, color='b', linewidth=2, label='original pdf')
            fit.power_law.plot_pdf(color='b', linestyle='--',
                                ax=fig2,
                                label=r"fit pdf ($\alpha={:.2f},\sigma={:.2f}$)".format(fit.alpha, fit.sigma))
            plt.axvline(fit.xmin, color='g', linestyle='--', label='xmin={}'.format(int(fit.xmin)))
            plt.xlabel('x')
            plt.ylabel('$p(x)$')
            plt.legend()
        except Exception as e:
            print(str(e))
        plt.savefig(filenamenoxmin)
        print(filenamenoxmin)
        plt.close(fig)

    if not os.path.exists(filenamexminfull) or not os.path.exists(filenamexminwindow) or force:
        fit = powerlaw.Fit(x, discrete=True)
        for o in (True, False):
            plt.clf()
            fig = plt.figure()
            try:
                fig2 = fit.plot_pdf(original_data=o, color='b', linewidth=2, label='original pdf')
                fit.power_law.plot_pdf(color='b', linestyle='--',
                                       ax=fig2,
                                       label=r"fit pdf ($\alpha={:.2f},\sigma={:.2f}$)".format(fit.alpha, fit.sigma))
                plt.axvline(fit.xmin, color='g', linestyle='--', label='xmin={}'.format(int(fit.xmin)))
                plt.xlabel('x')
                plt.ylabel('$p(x)$')
                plt.legend()
            except Exception as e:
                print(str(e))
            if o == True:
                plt.savefig(filenamexminfull)
                print(filenamexminfull)
            else:
                plt.savefig(filenamexminwindow)
                print(filenamexminwindow)
            plt.close(fig)
    return cat
