import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx

# Enable mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load FashionMNIST Dataset
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Optimized DataLoader
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=8)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=8)

# Dendritic Neuron with Optimizations
class DendriticNeuron(nn.Module):
    def __init__(self, input_size, num_dendrites):
        super(DendriticNeuron, self).__init__()
        self.dendritic_weights = nn.Parameter(torch.randn(num_dendrites, input_size, device=device))
        self.soma_weights = nn.Parameter(torch.randn(num_dendrites, device=device))
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, inputs):
        dendritic_outputs = self.activation(torch.matmul(inputs, self.dendritic_weights.T))
        return self.activation(torch.matmul(dendritic_outputs, self.soma_weights))

# Optimized DendriticANN
class DendriticANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers, dendrite_counts):
        super(DendriticANN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.dendrite_counts = dendrite_counts
        
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([
            nn.ModuleList([DendriticNeuron(hidden_size, dendrite_counts[layer_idx * hidden_size + neuron_idx]) for neuron_idx in range(hidden_size)])
            for layer_idx in range(num_hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.activation(self.input_layer(x))

        for layer in self.hidden_layers:
            layer_output = torch.zeros_like(x)
            for neuron_idx, neuron in enumerate(layer):
                layer_output[:, neuron_idx] = neuron(x).squeeze()
            x = self.activation(layer_output)

        return self.output_layer(x)

# Function to Evaluate Model
@torch.no_grad()
def evaluate_model(model, data_loader):
    model.eval()
    correct, total = 0, 0
    for inputs, labels in data_loader:
        inputs, labels = inputs.view(inputs.size(0), -1).to(device), labels.to(device)
        outputs = model(inputs)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return (correct / total) * 100
	
# Function to visualize the network using networkx
def visualize_network(model, generation):
    G = nx.DiGraph()
    
    # Add input layer nodes
    for i in range(model.input_size):
        G.add_node(f"Input_{i}", layer=0)
    
    # Add hidden layer nodes and edges
    for layer_idx, layer in enumerate(model.hidden_layers):
        for neuron_idx, neuron in enumerate(layer):
            node_name = f"Hidden_{layer_idx}_{neuron_idx}"
            G.add_node(node_name, layer=layer_idx + 1)
            
            # Connect to previous layer
            if layer_idx == 0:
                for input_idx in range(model.input_size):
                    G.add_edge(f"Input_{input_idx}", node_name)
            else:
                for prev_neuron_idx in range(model.hidden_size):
                    G.add_edge(f"Hidden_{layer_idx - 1}_{prev_neuron_idx}", node_name)
    
    # Add output layer nodes and edges
    for output_idx in range(model.output_size):
        node_name = f"Output_{output_idx}"
        G.add_node(node_name, layer=model.num_hidden_layers + 1)
        for prev_neuron_idx in range(model.hidden_size):
            G.add_edge(f"Hidden_{model.num_hidden_layers - 1}_{prev_neuron_idx}", node_name)
    
    # Draw the graph
    pos = nx.multipartite_layout(G, subset_key="layer")
    plt.figure(figsize=(120, 80))
    nx.draw(G, pos, with_labels=True, node_size=10, node_color="lightblue", font_size=5, font_weight="bold", width=0.01)
    plt.title(f"Best Network at Generation {generation}")
    plt.savefig(f"BestNetGen{generation}")

# Function to plot accuracy over generations
def plot_accuracy(best_accuracies, avg_accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(best_accuracies, label="Best Accuracy", marker="o")
    plt.plot(avg_accuracies, label="Average Accuracy", marker="x")
    plt.xlabel("Generation")
    plt.ylabel("Accuracy (%)")
    plt.title("Best and Average Accuracy Over Generations")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy.png")

# Function to plot total number of dendrites in the best network over epochs
def plot_dendrites_over_epochs(dendrite_counts_over_epochs):
    plt.figure(figsize=(10, 6))
    plt.plot(dendrite_counts_over_epochs, label="Total Dendrites", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Total Number of Dendrites")
    plt.title("Total Number of Dendrites in Best Network Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("dendrites_over_epochs.png")

class GeneticAlgorithm:
    def __init__(self, pop_size, input_size, hidden_size, output_size, num_hidden_layers, max_dendrites):
        self.pop_size = pop_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.max_dendrites = max_dendrites
        self.population = self._initialize_population()
        self.best_accuracies = []
        self.avg_accuracies = []
        self.dendrite_counts_over_epochs = []

    def _initialize_population(self):
        return [{
            "dendrite_counts": [random.randint(1, self.max_dendrites)
                                for _ in range(self.num_hidden_layers * self.hidden_size)],
            "state_dict": None  # Initialize with no weights
        } for _ in range(self.pop_size)]

    def adaptively_load_weights(self, model, saved_state_dict):
        model_state = model.state_dict()
    
        for name, param in model_state.items():
            if name not in saved_state_dict:
                continue
    
            saved_param = saved_state_dict[name]
            if param.shape == saved_param.shape:
                param.copy_(saved_param)
            elif saved_param.ndim == param.ndim and all(
                s == p or i == 0 for i, (s, p) in enumerate(zip(saved_param.shape, param.shape))
            ):
                # Case: Only first dimension (e.g., # dendrites) has changed
                min_shape = [min(s, p) for s, p in zip(saved_param.shape, param.shape)]
                slices = tuple(slice(0, m) for m in min_shape)
    
                # Copy over what's compatible
                param[slices] = saved_param[slices]
    
                # Randomize extra part if param is larger
                if any(p > s for s, p in zip(saved_param.shape, param.shape)):
                    with torch.no_grad():
                        rand_init = torch.randn_like(param)
                        param[slices] = saved_param[slices]
                        param.copy_(torch.where(
                            torch.isnan(param), rand_init, param
                        ))
            else:
                print(f"[WARN] Skipped: shape mismatch for {name} ({saved_param.shape} → {param.shape})")

    def evaluate_fitness(self, individual):
        model = DendriticANN(self.input_size, self.hidden_size, self.output_size,
                             self.num_hidden_layers, individual["dendrite_counts"]).to(device)

        # Load inherited weights if available
        if individual.get("state_dict"):
            self.adaptively_load_weights(model, individual["state_dict"])

        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=False)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0
        epochs = 3
        for epoch in range(epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.view(inputs.size(0), -1).to(device), labels.to(device)

                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            acc = evaluate_model(model, test_loader)
            scheduler.step(acc)
            best_val_acc = max(best_val_acc, acc)

        individual["fitness"] = best_val_acc
        individual["state_dict"] = model.state_dict()
        return best_val_acc

    def evolve(self, generations, elitism_count=2):
        for gen in range(generations):
            print(f"Generation {gen + 1}/{generations}")
            fitness_scores = [self.evaluate_fitness(ind) for ind in tqdm(self.population, desc="Evaluating", unit="ind")]
            sorted_population = sorted(self.population, key=lambda x: x["fitness"], reverse=True)

            self.best_accuracies.append(sorted_population[0]["fitness"])
            self.avg_accuracies.append(np.mean([ind["fitness"] for ind in sorted_population]))
            self.dendrite_counts_over_epochs.append(sum(sorted_population[0]["dendrite_counts"]))

            next_gen = sorted_population[:elitism_count]
            top_80 = sorted_population[:int(self.pop_size * 0.8)]

            while len(next_gen) < self.pop_size:
                parent1, parent2 = random.choices(top_80, k=2)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                next_gen.append(child)

            self.population = next_gen
            print(f"Best Accuracy: {max(fitness_scores):.2f}%")

        plot_accuracy(self.best_accuracies, self.avg_accuracies)
        plot_dendrites_over_epochs(self.dendrite_counts_over_epochs)

    def _crossover(self, parent1, parent2):
        # Average dendrite counts
        dendrites = [(p1 + p2) // 2 for p1, p2 in zip(parent1["dendrite_counts"], parent2["dendrite_counts"])]
        child = {"dendrite_counts": dendrites, "state_dict": None}
    
        # Merge parent weights if available
        if parent1.get("state_dict") and parent2.get("state_dict"):
            blended = {}
            for k in parent1["state_dict"]:
                if k in parent2["state_dict"]:
                    # Try to average overlapping parts
                    t1, t2 = parent1["state_dict"][k], parent2["state_dict"][k]
                    try:
                        blended[k] = (t1 + t2) / 2
                    except RuntimeError:
                        # Fallback: Just take from the fitter parent (will be adapted later)
                        blended[k] = t1
            child["state_dict"] = blended if blended else None
    
        return child

    def _mutate(self, individual):
        for i in range(len(individual["dendrite_counts"])):
            if random.random() < 0.1:
                # Bias toward reducing size (encouraging simplicity)
                delta = random.choices([-1, 0, 1], weights=[0.5, 0.3, 0.2])[0]
                individual["dendrite_counts"][i] = max(1, min(individual["dendrite_counts"][i] + delta, self.max_dendrites))
        return individual

# Run Genetic Algorithm
ga = GeneticAlgorithm(pop_size=50, input_size=784, hidden_size=128, output_size=10, num_hidden_layers=1, max_dendrites=10)
ga.evolve(generations=20)