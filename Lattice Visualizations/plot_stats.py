from matplotlib import pyplot as plt


def plot_beta_results(betas, errors, best_beta):
    plt.figure(figsize=(10, 6))
    plt.plot(betas, errors, marker='o')
    plt.title('Distortion vs. Beta')
    plt.xlabel('Beta')
    plt.ylabel('Average Distortion (Error)')
    plt.grid()
    plt.axvline(x=best_beta, color='r', linestyle='--', label=f'Best Beta: {best_beta:.2f}')
    plt.legend()
    plt.show()


def plot_q_results(lattice_name, q_values, average_errors):
    plt.figure(figsize=(12, 6))
    plt.plot(q_values, average_errors, marker='o', label='Average Error')
    plt.title(f'Average Distortion vs. q for {lattice_name} Lattice')
    plt.xlabel('q')
    plt.ylabel('Average Distortion (Error)')
    plt.grid()

    for q in q_values:
        plt.axvline(x=q, color='gray', linestyle='--', alpha=0.5)

    plt.legend()
    plt.show()
