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
from copy import deepcopy
import time
from datetime import timedelta

# Enable mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data Transformations with augmentation
transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load FashionMNIST Dataset
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

# Split train into train and validation (80/20 split)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Test set remains unchanged
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Create DataLoaders for all three sets
batch_size = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=8)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=8)

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
def plot_accuracy(best_accuracies, avg_accuracies, filename="accuracy.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(best_accuracies, label="Best Accuracy", marker="o")
    plt.plot(avg_accuracies, label="Average Accuracy", marker="x")
    plt.xlabel("Generation")
    plt.ylabel("Accuracy (%)")
    plt.title("Best and Average Accuracy Over Generations")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)

def plot_loss(loss, filename="loss.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(loss, label="Loss", marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Loss")
    plt.title("Loss Over Generations")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)

def plot_fitness(best_fitness, avg_fitness, filename="fitness.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness, label="Best Fitness", marker="o")
    plt.plot(avg_fitness, label="Average Fitness", marker="x")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Best and Average Fitness Over Generations")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)

# Function to plot total number of dendrites in the best network over epochs
def plot_dendrites_over_epochs(dendrite_counts_over_epochs):
    plt.figure(figsize=(10, 6))
    plt.plot(dendrite_counts_over_epochs, label="Total Dendrites", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Total Number of Dendrites")
    plt.title("Total Number of Dendrites in Best Network Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("gamnist/dendrites_over_epochs.png")

class DendriticNeuron(nn.Module):
    def __init__(self, input_size, num_dendrites):
        super(DendriticNeuron, self).__init__()
        self.dendritic_weights = nn.Parameter(torch.randn(num_dendrites, input_size, device=device) * 0.1)
        self.soma_weights = nn.Parameter(torch.randn(num_dendrites, device=device) * 0.1)
        self.activation = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.2)
        
        # Track dendrite activity
        self.register_buffer('dendrite_activity', torch.zeros(num_dendrites))
        
    def forward(self, inputs):
        inputs = nn.functional.normalize(inputs, p=2, dim=1)
        dendritic_outputs = self.activation(torch.matmul(inputs, self.dendritic_weights.T))
        
        with torch.no_grad():
            self.dendrite_activity = 0.9 * self.dendrite_activity + 0.1 * dendritic_outputs.mean(dim=0).abs()
            self.dendrite_activity = self.dendrite_activity.to(device)
            
        dendritic_outputs = self.dropout(dendritic_outputs)
        return self.activation(torch.matmul(dendritic_outputs, self.soma_weights))

class DendriticANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers, dendrite_counts):
        super(DendriticANN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.dendrite_counts = dendrite_counts
        
        self.input_layer = nn.Linear(input_size, hidden_size)
        nn.init.xavier_uniform_(self.input_layer.weight, gain=nn.init.calculate_gain('leaky_relu', 0.01))
        
        self.hidden_layers = nn.ModuleList([
            nn.ModuleList([DendriticNeuron(hidden_size, dendrite_counts[layer_idx * hidden_size + neuron_idx]) 
                          for neuron_idx in range(hidden_size)])
            for layer_idx in range(num_hidden_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=nn.init.calculate_gain('linear'))
        self.activation = nn.LeakyReLU(0.01)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.activation(self.input_layer(x))
        
        # Batch Normalization after input layer
        x = nn.BatchNorm1d(self.hidden_size).to(device)(x)
        
        for layer in self.hidden_layers:
            layer_output = torch.zeros_like(x)
            for neuron_idx, neuron in enumerate(layer):
                layer_output[:, neuron_idx] = neuron(x).squeeze()
            x = self.activation(layer_output)
            
            # Batch Normalization after each hidden layer
            x = nn.BatchNorm1d(self.hidden_size).to(device)(x)
        
        return self.output_layer(x)

class GeneticAlgorithm:
    def __init__(self, pop_size, input_size, hidden_size, output_size, num_hidden_layers, max_dendrites):
        self.pop_size = pop_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.max_dendrites = max_dendrites
        self.population = self._initialize_population()
        self.best_train_accuracies = []
        self.avg_train_accuracies = []
        self.best_test_accuracies = []
        self.avg_test_accuracies = []
        self.best_val_accuracies = []
        self.avg_val_accuracies = []
        self.best_fitness = []
        self.avg_fitness = []
        self.best_loss = []
        self.avg_loss = []
        self.best_val_loss = []
        self.avg_val_loss = []
        self.dendrite_counts_over_epochs = []
        self.best_model_state = None
        self.start_time = None
        self.generation_times = []
    
    def _initialize_population(self):
        return [{
            "dendrite_counts": [random.randint(1, self.max_dendrites)
                                for _ in range(self.num_hidden_layers * self.hidden_size)],
            "state_dict": None,
            "fitness": 0
        } for _ in range(self.pop_size)]
    
    def robust_weight_inheritance(self, model, parent_state_dict):
        """Combined approach: shape adaptation + activity-based selection"""
        model_state = model.state_dict()
        new_state_dict = {}
        
        for name, param in model_state.items():
            if name in parent_state_dict:
                parent_param = parent_state_dict[name]
    
                # Skip if parent parameter is None (for some buffers)
                if parent_param is None:
                    continue
                    
                # Handle 1D tensors (biases, soma_weights)
                if parent_param.dim() == 1:
                    min_len = min(param.shape[0], parent_param.shape[0])
                    new_param = torch.zeros_like(param)
                    new_param[:min_len] = parent_param[:min_len]
                    new_state_dict[name] = new_param
                    continue
                    
                # Handle 2D tensors (weights)
                try:
                    activity = parent_param.norm(dim=1)
                    min_dendrites = min(param.shape[0], parent_param.shape[0])
                    top_indices = torch.argsort(activity, descending=True)[:min_dendrites]
                    
                    # Case 1: Exact shape match
                    if param.shape == parent_param.shape:
                        new_state_dict[name] = parent_param.clone()
                    
                    # Case 2: Dendritic weights - inherit most active first
                    elif 'dendritic_weights' in name:
                        new_weights = torch.zeros_like(param)
                        new_weights[:min_dendrites] = parent_param[top_indices]
                        new_state_dict[name] = new_weights
                    
                    # Case 3: Soma weights - match dendrite selection
                    elif 'soma_weights' in name:
                        new_weights = torch.zeros_like(param)
                        new_weights[:min_dendrites] = parent_param[top_indices]
                        new_state_dict[name] = new_weights
                    
                    # Case 4: Linear layers
                    else:
                        min_out = min(param.shape[0], parent_param.shape[0])
                        min_in = min(param.shape[1], parent_param.shape[1])
                        new_weights = torch.zeros_like(param)
                        new_weights[:min_out, :min_in] = parent_param[:min_out, :min_in]
                        
                        if param.shape[0] > parent_param.shape[0] or param.shape[1] > parent_param.shape[1]:
                            scale = parent_param.std().item()
                            noise = torch.randn_like(param) * scale * 0.1
                            mask = (new_weights == 0).float()
                            new_weights = new_weights + noise * mask
                        
                        new_state_dict[name] = new_weights
                except Exception as e:
                    print(f"Error processing {name}: {e}")
                    new_state_dict[name] = param.clone()
        
        model.load_state_dict(new_state_dict, strict=False)
        for param in model.parameters():
            param.data = param.data * 0.99
    
    def plot_dendrite_distribution(self, model, generation):
        """Visualize active vs total dendrites"""
        total = []
        active = []
        for neuron in model.hidden_layers[0]:
            total.append(neuron.dendritic_weights.shape[0])
            active.append((neuron.dendrite_activity > 0.1).sum().item())
        
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(total)), total, alpha=0.5, label='Total Dendrites')
        plt.bar(range(len(active)), active, alpha=0.8, label='Active Dendrites')
        plt.xlabel('Neuron Index')
        plt.ylabel('Count')
        plt.legend()
        plt.title(f'Dendrite Distribution (Gen {generation})')
        plt.savefig(f'gamnist/dendrite_distribution_gen{generation}.png')
        plt.close()
    
    def evaluate_fitness(self, individual, epochs=5):
        model = DendriticANN(self.input_size, self.hidden_size, self.output_size,
                           self.num_hidden_layers, individual["dendrite_counts"]).to(device)
        
        if individual.get("state_dict"):
            self.robust_weight_inheritance(model, individual["state_dict"])
        
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1, factor=0.5, verbose=True)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        loss = 0
        val_loss = 0
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

            # Step the scheduler after each epoch based on the validation loss
            val_loss = evaluate_model(model, val_loader)
            scheduler.step(val_loss)
            
        train_acc = evaluate_model(model, train_loader)
        acc = evaluate_model(model, test_loader)
        val_acc = evaluate_model(model, val_loader)
        
        total_dendrites = sum(individual["dendrite_counts"])
        max_possible = self.hidden_size * self.max_dendrites
        dendrite_penalty = 0.01 * (total_dendrites / max_possible)
        
        individual["fitness"] = val_acc * (1 - dendrite_penalty)
        individual["train_acc"] = train_acc
        individual["val_acc"] = val_acc
        individual["test_acc"] = acc
        individual["loss"] = criterion(outputs, labels).item()
        individual["val_loss"] = val_loss
        individual["state_dict"] = model.state_dict()
        
        if self.best_model_state is None or val_acc > self.best_model_state["fitness"]:
            self.best_model_state = {
                "state_dict": deepcopy(model.state_dict()),
                "fitness": val_acc,
                "dendrite_counts": deepcopy(individual["dendrite_counts"])
            }
            self.plot_dendrite_distribution(model, len(self.best_val_accuracies))
        
        return individual["fitness"]
    
    def _enhanced_crossover(self, parent1, parent2):
        strategy = random.choices(
            ["uniform", "average", "single_point"],
            weights=[0.4, 0.4, 0.2],
            k=1
        )[0]
        
        if strategy == "uniform":
            dendrites = [random.choice([p1, p2]) for p1, p2 in zip(
                parent1["dendrite_counts"], parent2["dendrite_counts"])]
        elif strategy == "average":
            avg = [(p1 + p2) / 2 for p1, p2 in zip(
                parent1["dendrite_counts"], parent2["dendrite_counts"])]
            noise = [random.gauss(0, 0.5) for _ in avg]
            dendrites = [max(1, min(int(round(a + n)), self.max_dendrites)) 
                        for a, n in zip(avg, noise)]
        else:  # single_point
            point = random.randint(1, len(parent1["dendrite_counts"]) - 1)
            dendrites = parent1["dendrite_counts"][:point] + parent2["dendrite_counts"][point:]
        
        child = {"dendrite_counts": dendrites, "state_dict": None, "fitness": 0}
        
        if parent1.get("state_dict") and parent2.get("state_dict"):
            blended = {}
            for k in parent1["state_dict"]:
                if k in parent2["state_dict"] and parent1["state_dict"][k].shape == parent2["state_dict"][k].shape:
                    alpha = random.random()
                    blended[k] = alpha * parent1["state_dict"][k] + (1 - alpha) * parent2["state_dict"][k]
            child["state_dict"] = blended
        
        return child
    
    def _adaptive_mutate(self, individual, current_gen, total_gens):
        min_mutation_rate = 0.05
        max_mutation_rate = 0.3
        progress = current_gen / total_gens
        current_rate = max_mutation_rate * (1 - progress) + min_mutation_rate * progress
        
        model = DendriticANN(self.input_size, self.hidden_size, self.output_size,
                           self.num_hidden_layers, individual["dendrite_counts"]).to(device)
        
        if individual.get("state_dict"):
            self.robust_weight_inheritance(model, individual["state_dict"])
        
        with torch.no_grad():
            batch = next(iter(train_loader))
            _ = model(batch[0].view(batch[0].size(0), -1).to(device))
        
        for neuron_idx in range(len(individual["dendrite_counts"])):
            neuron = model.hidden_layers[0][neuron_idx]
            activity = (neuron.dendrite_activity + 1e-6) / (neuron.dendrite_activity.sum() + 1e-6)
            
            if random.random() < current_rate:
                current = individual["dendrite_counts"][neuron_idx]
                if activity.mean() < 0.05 and current > 1:
                    individual["dendrite_counts"][neuron_idx] = current - 1
                elif activity.mean() > 0.3 and current < self.max_dendrites:
                    individual["dendrite_counts"][neuron_idx] = current + 1
                else:
                    delta = random.randint(-1, 1)
                    individual["dendrite_counts"][neuron_idx] = max(1, min(current + delta, self.max_dendrites))
        
        return individual
    
    def evolve(self, generations, elitism_count=5):
        self.start_time = time.time()
        print(f"\nStarting evolution for {generations} generations...")
        
        for gen in range(generations):
            gen_start = time.time()
            print(f"\nGeneration {gen + 1}/{generations}")
            
            # Evaluate population
            fitness_scores = []
            for ind in tqdm(self.population, desc="Evaluating"):
                fitness = self.evaluate_fitness(ind)
                fitness_scores.append(fitness)
            
            # Sort and track metrics
            sorted_pop = sorted(self.population, key=lambda x: x["loss"], reverse=False)
            self.best_loss.append(sorted_pop[0]["loss"])
            self.avg_loss.append(np.mean([ind["loss"] for ind in sorted_pop]))

            sorted_pop = sorted(self.population, key=lambda x: x["val_loss"], reverse=False)
            self.best_val_loss.append(sorted_pop[0]["val_loss"])
            self.avg_val_loss.append(np.mean([ind["val_loss"] for ind in sorted_pop]))
            
            sorted_pop = sorted(self.population, key=lambda x: x["train_acc"], reverse=True)
            self.best_train_accuracies.append(sorted_pop[0]["train_acc"])
            self.avg_train_accuracies.append(np.mean([ind["train_acc"] for ind in sorted_pop]))
            
            sorted_pop = sorted(self.population, key=lambda x: x["test_acc"], reverse=True)
            self.best_test_accuracies.append(sorted_pop[0]["test_acc"])
            self.avg_test_accuracies.append(np.mean([ind["test_acc"] for ind in sorted_pop]))

            sorted_pop = sorted(self.population, key=lambda x: x["val_acc"], reverse=True)
            self.best_val_accuracies.append(sorted_pop[0]["val_acc"])
            self.avg_val_accuracies.append(np.mean([ind["val_acc"] for ind in sorted_pop]))

            sorted_pop = sorted(self.population, key=lambda x: x["fitness"], reverse=True)
            self.best_fitness.append(sorted_pop[0]["fitness"])
            self.avg_fitness.append(np.mean([ind["fitness"] for ind in sorted_pop]))
            self.dendrite_counts_over_epochs.append(sum(sorted_pop[0]["dendrite_counts"]))
            
            # Create next generation
            next_gen = sorted_pop[:elitism_count]
            top_80 = sorted_pop[:int(self.pop_size * 0.8)]
            
            # Produce offspring
            while len(next_gen) < self.pop_size:
                parents = random.choices(top_80, k=2)
                child = self._enhanced_crossover(*parents)
                child = self._adaptive_mutate(child, gen, generations)
                next_gen.append(child)
            
            self.population = next_gen
            
            # Timing and progress
            gen_time = time.time() - gen_start
            self.generation_times.append(gen_time)
            avg_time = np.mean(self.generation_times)
            remaining = timedelta(seconds=round(avg_time * (generations - gen - 1)))
            
            print(f"Best Accuracy: {sorted_pop[0]['fitness']:.2f}%")
            print(f"Generation Time: {gen_time:.1f}s | Est. Remaining: {remaining}")
            
            # Memory management
            if gen % 5 == 0:
                torch.cuda.empty_cache()
        
        # Final results
        total_time = timedelta(seconds=round(time.time() - self.start_time))
        print(f"\nEvolution completed in {total_time}")
        print(f"Final Best Val Accuracy: {max(self.best_val_accuracies):.2f}%")
        print(f"Final Best Test Accuracy: {max(self.best_test_accuracies):.2f}%")
        
        # Save visualizations
        plot_accuracy(self.best_train_accuracies, self.avg_train_accuracies, "gamnist/test_acc.png")
        plot_accuracy(self.best_test_accuracies, self.avg_test_accuracies, "gamnist/test_acc.png")
        plot_accuracy(self.best_val_accuracies, self.avg_val_accuracies, "gamnist/val.png")
        plot_fitness(self.best_fitness, self.avg_fitness, "gamnist/fitness.png")
        plot_loss(self.best_loss, self.avg_loss, "gamnist/loss.png")
        plot_loss(self.best_val_loss, self.avg_val_loss, "gamnist/val_loss.png")
        plot_dendrites_over_epochs(self.dendrite_counts_over_epochs)

        torch.save(self.best_model_state["state_dict"], "best_model.pth")
        
        # Visualize best network
        # best_model = DendriticANN(self.input_size, self.hidden_size, self.output_size,
        #                         self.num_hidden_layers, self.best_model_state["dendrite_counts"]).to(device)
        # best_model.load_state_dict(self.best_model_state["state_dict"])
        # visualize_network(best_model, generations)

ga = GeneticAlgorithm(pop_size=50, input_size=784, hidden_size=128, 
                     output_size=10, num_hidden_layers=1, max_dendrites=10)
ga.evolve(generations=20)
