import os
import matplotlib.pyplot as plt

def plot_graphs_embedder(stats_list, vector_list, time_descriptions, memory_descriptions):
    # Extract data for plotting
    # num_times = len(stats_list[0])
    # num_memories = len(stats_list[1])

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plotting the graphs for Time
    for i, desc in enumerate(time_descriptions):
        times = [stats[0][i] for stats in stats_list]
        axs[0].plot(vector_list, times, marker='o', label=desc)
    axs[0].set_title('Embeddings Classification Time')
    axs[0].set_xlabel('Vector Size')
    axs[0].set_ylabel('Time (sec)')
    axs[0].legend()

    # Plotting the graphs for Memory
    for i, desc in enumerate(memory_descriptions):
        memories = [stats[1][i] for stats in stats_list]
        axs[1].plot(vector_list, memories, marker='o', label=desc)
    axs[1].set_title('Embeddings Classification Memory Usage')
    axs[1].set_xlabel('Vector Size')
    axs[1].set_ylabel('Memory (MB)')
    axs[1].legend()

    fig.tight_layout()

    # Create directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Save the plot in SVG format
    plt.savefig('plots/embeddings_stats_plots.svg', format='svg', dpi=300)

    # Save the plot in high-resolution PNG format
    plt.savefig('plots/embeddings_stats_plots.png', format='png', dpi=300)

    # plt.show()


def plot_graphs_classifier(stats_list, vector_list, time_descriptions, memory_descriptions):
    # Extract data for plotting
    # num_times = len(stats_list[0])
    # num_memories = len(stats_list[1])

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plotting the graphs for Time
    for i, desc in enumerate(time_descriptions):
        times = [stats[0][i] for stats in stats_list]
        axs[0].plot(vector_list, times, marker='o', label=desc)
    axs[0].set_title('Embeddings Classification Time')
    axs[0].set_xlabel('Vector Size')
    axs[0].set_ylabel('Time (sec)')
    axs[0].legend()

    # Plotting the graphs for Memory
    for i, desc in enumerate(memory_descriptions):
        memories = [stats[1][i] for stats in stats_list]
        axs[1].plot(vector_list, memories, marker='o', label=desc)
    axs[1].set_title('Embeddings Classification Memory Usage')
    axs[1].set_xlabel('Vector Size')
    axs[1].set_ylabel('Memory (MB)')
    axs[1].legend()

    fig.tight_layout()

    # Create directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Save the plot in SVG format
    plt.savefig('plots/embeddings_stats_classifier_plots.svg', format='svg', dpi=300)

    # Save the plot in high-resolution PNG format
    plt.savefig('plots/embeddings_stats_classifier_plots.png', format='png', dpi=300)

    # plt.show()

# if __name__ == "__main__":
#     # Example data for plot_graphs_embedder
#     stats_list_embedder = [
#         [(10.2, 15.5, 20.1), (100, 120, 130)],
#         [(8.5, 12.9, 18.7), (90, 110, 115)],
#         [(7.1, 10.8, 15.2), (80, 100, 105)],
#         [(6.2, 9.5, 13.8), (75, 95, 100)],
#         [(5.5, 8.7, 12.1), (70, 90, 95)],
#         [(4.8, 7.9, 10.5), (65, 85, 90)],
#         [(3.9, 6.5, 9.1), (60, 80, 85)],
#         [(3.2, 5.8, 7.8), (55, 75, 80)]
#     ]
#     vector_list = [768, 512, 256, 128, 64, 32, 15, 5]
#     time_descriptions_embedder = [
#         "FastText Training Time per Flow",
#         "FastText Embeddings Creation Time per Flow",
#         "BERT Embeddings Creation Time per Flow"
#     ]
#     memory_descriptions_embedder = [
#         "FastText Training Memory Usage per Flow",
#         "FastText Embeddings Creation Memory Usage per Flow",
#         "BERT Embeddings Creation Memory Usage per Flow"
#     ]

#     plot_graphs_embedder(stats_list_embedder, vector_list, time_descriptions_embedder, memory_descriptions_embedder)

#     # Example data for plot_graphs_classifier
#     stats_list_classifier = [
#         [(10.2, 15.5), (100, 120)],
#         [(8.5, 12.9), (90, 110)],
#         [(7.1, 10.8), (80, 100)],
#         [(6.2, 9.5), (75, 95)],
#         [(5.5, 8.7), (70, 90)],
#         [(4.8, 7.9), (65, 85)],
#         [(3.9, 6.5), (60, 80)],
#         [(3.2, 5.8), (55, 75)]
#     ]
#     vector_list = [768, 512, 256, 128, 64, 32, 15, 5]
#     time_descriptions_classifier = [
#         "FastText Embeddings classification Time per Flow",
#         "BERT Embeddings classification Time per Flow"
#     ]
#     memory_descriptions_classifier = [
#         "FastText Embeddings classification Memory Usage per Flow",
#         "BERT Embeddings classification Memory Usage per Flow"
#     ]

#     plot_graphs_classifier(stats_list_classifier, vector_list, time_descriptions_classifier, memory_descriptions_classifier)
