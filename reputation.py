import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy


# ===========================================================
# REPUTATION MANAGEMENT SYSTEM
# ===========================================================

class ReputationManager:
    """
    Advanced multi-method reputation management for federated learning.
    Implements 7 different reputation calculation methods with validation.
    """

    def __init__(self, num_clients, alpha_beta=2.0, lambda_decay=0.1):
        self.num_clients = num_clients
        self.alpha_beta = alpha_beta  # Beta distribution parameter
        self.lambda_decay = lambda_decay  # Exponential decay rate

        # Historical tracking
        self.reputation_history = [[] for _ in range(num_clients)]
        self.performance_history = [[] for _ in range(num_clients)]
        self.validation_improvements = [[] for _ in range(num_clients)]

        # Beta reputation tracking (success/failure)
        self.successes = np.ones(num_clients)  # Start with 1 to avoid division by zero
        self.failures = np.ones(num_clients)

        # Consistency tracking
        self.previous_predictions = [None] * num_clients

        # Baseline performance
        self.baseline_performance = None

    def calculate_weighted_average(self, client_id, current_score, window=5):
        """
        Method 1: Weighted Average - Smooth running average
        Recent scores get higher weights.
        """
        history = self.reputation_history[client_id]
        if len(history) == 0:
            return current_score

        recent = history[-window:] + [current_score]
        weights = np.exp(np.linspace(0, 1, len(recent)))
        weights = weights / weights.sum()

        return np.sum(np.array(recent) * weights)

    def calculate_consistency(self, client_index, y_pred_test):
        """
        Calcola la consistenza delle predizioni del client
        rispetto al round precedente.
        """

        previous_predictions = self.previous_predictions[client_index]

        if previous_predictions is None:
            consistency = 1.0
        else:
            consistency = np.mean(
                y_pred_test == previous_predictions
            )

        # Salviamo le predizioni per il prossimo round
        self.previous_predictions[client_index] = y_pred_test.copy()

        return float(consistency)

    def calculate_entropy_reputation(self, prediction_probs, n_classes):
        """
            Method 6: Entropy-based - Measure uncertainty/consistency
            Best for: Data consistency checks
            Lower entropy = more confident predictions = higher reputation
        """

        # Calculate entropy of prediction distribution
        pred_entropy = entropy(prediction_probs.T + 1e-10)  # Add small value to avoid log(0)
        avg_entropy = np.mean(pred_entropy)

        # Normalize entropy (max entropy = log(n_classes))
        max_entropy = np.log(n_classes)
        normalized_entropy = avg_entropy / max_entropy

        # Convert to reputation (low entropy = high reputation)
        reputation = 1.0 - normalized_entropy

        return reputation

    def calculate_beta_reputation(self, client_id):
        """
        Method 2: Beta Reputation - Probabilistic success/failure history
        Best for: Wireless/IoT FL with connection reliability concerns
        """
        alpha = self.successes[client_id]
        beta = self.failures[client_id]

        # Expected value of Beta distribution
        reputation = alpha / (alpha + beta)

        # Uncertainty (variance) - lower is better
        variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))

        # Combine reputation with confidence (lower variance = higher confidence)
        confidence_factor = 1.0 / (1.0 + variance * 10)

        return reputation * confidence_factor

    def calculate_fuzzy_trust(self, client_id, accuracy, consistency, plausibility):
        """
        Method 3: Fuzzy Trust - Multi-factor combination via fuzzy rules
        Best for: Smart contracts and complex decision making
        """
        # Fuzzy membership functions
        def triangular(x, a, b, c):

            if x < a or x > c:
                return 0.0
            if x == b:
                return 1.0
            if a == b:
                return (c - x) / (c - b)
            if b == c:
                return (x - a) / (b - a)
            if x < b:
                return (x - a) / (b - a)
            return (c - x) / (c - b)

        # Define fuzzy sets for each metric
        # Accuracy fuzzy sets: low, medium, high
        acc_low = triangular(accuracy, 0, 0, 0.5)
        acc_med = triangular(accuracy, 0.3, 0.5, 0.7)
        acc_high = triangular(accuracy, 0.6, 1.0, 1.0)

        # Consistency fuzzy sets
        cons_low = triangular(consistency, 0, 0, 0.5)
        cons_high = triangular(consistency, 0.5, 1.0, 1.0)

        # Plausibility fuzzy sets
        plaus_low = triangular(plausibility, 0, 0, 0.5)
        plaus_high = triangular(plausibility, 0.5, 1.0, 1.0)

        # Fuzzy rules (simplified)
        trust = 0.0
        trust += min(acc_high, cons_high, plaus_high) * 1.0  # All high -> high trust
        trust += min(acc_med, cons_high, plaus_high) * 0.7   # Med acc, high others -> med trust
        trust += min(acc_low, cons_high, plaus_high) * 0.3   # Low acc -> low trust

        return min(trust, 1.0)

    def calculate_tanh_utility(self, client_id, validation_improvement):
        """
        Method 4: Tanh Utility - Smooth nonlinear stable updates
        Best for: FL with contribution metrics
        Uses hyperbolic tangent for smooth bounded reputation updates
        """
        # Tanh transforms (-inf, inf) to (-1, 1)
        # Shift to (0, 1) range
        utility = (np.tanh(validation_improvement * 5) + 1) / 2

        # Combine with historical reputation
        if len(self.reputation_history[client_id]) > 0:
            historical = np.mean(self.reputation_history[client_id][-3:])
            utility = 0.7 * utility + 0.3 * historical

        return utility

    def calculate_exponential_decay(self, client_id, current_score):
        """
        Method 5: Exponential Decay - Recent activity priority
        Best for: Real-time systems where recent performance matters most
        """
        history = self.reputation_history[client_id]
        if len(history) == 0:
            return current_score

        # Apply exponential decay to historical scores
        decayed_scores = []
        for i, score in enumerate(history):
            time_distance = len(history) - i
            decayed_score = score * np.exp(-self.lambda_decay * time_distance)
            decayed_scores.append(decayed_score)

        # Combine with current score (highest weight)
        all_scores = decayed_scores + [current_score]
        weights = np.exp(-self.lambda_decay * np.arange(len(all_scores)-1, -1, -1))
        weights = weights / weights.sum()

        return np.sum(np.array(all_scores) * weights)


    def cosine_similarity_reputation(self, client_index, client_updates):
        flat_updates = [
            np.concatenate([
                layer.flatten()
                for layer in update
            ])
            for update in client_updates
        ]

        similarity_matrix = cosine_similarity(flat_updates)

        other_similarities = np.delete(
            similarity_matrix[client_index],
            client_index
        )

        score = np.mean(other_similarities)

        # mapping [-1, 1] → [0, 1]
        score = (score + 1) / 2

        return score


    def calculate_validation_improvement(self, client_id, current_performance):
        """
        Calculate improvement over baseline and previous rounds.
        """
        if self.baseline_performance is None:
            return 0.0

        # Improvement over baseline
        baseline_improvement = current_performance - self.baseline_performance

        # Improvement over previous round
        if len(self.performance_history[client_id]) > 0:
            previous_performance = self.performance_history[client_id][-1]
            recent_improvement = current_performance - previous_performance
        else:
            recent_improvement = 0.0

        # Combined improvement metric
        improvement = 0.6 * baseline_improvement + 0.4 * recent_improvement

        return improvement

    def calculate_plausibility_score(self, client_id, accuracy, f1_score, global_avg_accuracy):
        """
        Check if client performance is plausible (not too good or too bad).
        Detects potential poisoning or faulty clients.
        """
        # Check if performance is within reasonable bounds
        deviation = abs(accuracy - global_avg_accuracy)

        # Plausibility decreases with deviation
        if deviation < 0.1:
            plausibility = 1.0
        elif deviation < 0.2:
            plausibility = 0.8
        elif deviation < 0.3:
            plausibility = 0.5
        else:
            plausibility = 0.2

        # Also check if metrics are consistent (accuracy and f1 should be similar)
        metric_consistency = 1.0 - abs(accuracy - f1_score)

        # Combined plausibility
        return 0.7 * plausibility + 0.3 * metric_consistency

    def update_client_reputation(self, client_id, metrics, client_index, client_updates, global_avg_accuracy, y_pred_test, p_test, current_round):
        """
        Master function: Calculate comprehensive reputation using all methods.
        """
        accuracy = metrics['accuracy']
        f1 = metrics['f1']

        # Calculate validation improvement
        validation_improvement = self.calculate_validation_improvement(client_index, accuracy)
        self.performance_history[client_index].append(accuracy)
        self.validation_improvements[client_index].append(validation_improvement)

        consistency = self.calculate_consistency(client_index,y_pred_test)

        entropy = self.calculate_entropy_reputation(p_test,p_test.shape[1])

        # Calculate global average for plausibility
        plausibility = self.calculate_plausibility_score(client_index, accuracy, f1, global_avg_accuracy)

        # Update success/failure for Beta reputation
        if accuracy > 0.7:  # Threshold for success
            self.successes[client_index] += 1
        else:
            self.failures[client_index] += 1

        # Calculate all reputation scores
        reputation_scores = {
            'weighted_avg': self.calculate_weighted_average(client_index, accuracy),
            'beta_reputation': self.calculate_beta_reputation(client_index),
            'fuzzy_trust': self.calculate_fuzzy_trust(client_index, accuracy, consistency, plausibility),
            'tanh_utility': self.calculate_tanh_utility(client_index, validation_improvement),
            'exponential_decay': self.calculate_exponential_decay(client_index, accuracy),
            'entropy_based': entropy,
            'cosine_similarity': self.cosine_similarity_reputation(client_index, client_updates),
            'consistency': consistency,
            'plausibility': plausibility
        }

        # Calculate combined reputation (weighted average of all methods)
        weights = {
            'weighted_avg': 0.15,
            'beta_reputation': 0.15,
            'fuzzy_trust': 0.15,
            'tanh_utility': 0.15,
            'exponential_decay': 0.10,
            'entropy_based': 0.10,
            'cosine_similarity': 0.15,
            'consistency': 0.025,
            'plausibility': 0.025
        }

        combined_reputation = sum(
            reputation_scores[k] * weights[k]
            for k in weights.keys()
        )

        # Ensure reputation is in [0, 1]
        combined_reputation = np.clip(combined_reputation, 0, 1)

        # Analisi del contributo delle singole metriche
        weighted_contributions = {
            k: reputation_scores[k] * weights[k]
            for k in weights.keys()
        }

        # Penalizzazione rispetto al valore ideale (= 1)
        penalties = {
            k: weights[k] * (1 - reputation_scores[k])
            for k in weights.keys()
        }

        # Ordina le metriche dalla più penalizzante alla meno penalizzante
        sorted_penalties = sorted(
            penalties.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Numero massimo di metriche da riportare = metà delle metriche totali
        max_negative_metrics = len(sorted_penalties) // 2

        # Penalizzazione peggiore
        max_penalty = sorted_penalties[0][1]

        penalty_threshold = max_penalty * (1- 0.3)

        negative_metrics = []

        for metric, penalty in sorted_penalties:

            # massimo numero di metriche raggiunto
            if len(negative_metrics) >= max_negative_metrics:
                break

            # metrica troppo distante dalla peggiore
            if penalty < penalty_threshold:
                break

            negative_metrics.append({
                'metric': metric,
                'value': reputation_scores[metric],
                'weight': weights[metric],
                'penalty': penalty
            })

        reputation_analysis = {
            'negative_metrics': negative_metrics
        }

        # Store in history
        self.reputation_history[client_index].append(combined_reputation)

        reputation_scores['combined'] = combined_reputation

        return reputation_scores, reputation_analysis

    def get_aggregation_weights(self, round_reputations):
        """
        Calculate aggregation weights based on reputation scores.
        Clients with higher reputation get higher weights in FedAvg.
        """
        reputations = np.array([r['combined'] for r in round_reputations])

        # Apply softmax to reputation scores for smooth weighting
        exp_reputations = np.exp(reputations * 2)  # Scale factor = 2
        weights = exp_reputations / exp_reputations.sum()

        return weights

    def should_include_client(self, client_reputation, threshold=0.3):
        """
        Decide if a client should be included in aggregation.
        Clients below threshold are excluded (potential malicious clients).
        """
        return client_reputation['combined'] >= threshold
