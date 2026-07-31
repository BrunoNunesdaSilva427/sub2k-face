

import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pc"))

import numpy as np
from synthetic_faces import generate_dataset
from pca_eigenfaces import EigenfaceModel


def euclidean_int8(a, b):

    diff = a.astype(np.int32) - b.astype(np.int32)
    return np.sqrt(np.sum(diff * diff))


def run_experiment(num_enrolled=5, num_impostor_identities=4,
                    enroll_samples=5, query_samples=10, n_components=24,
                    seed=42):
    total_identities = num_enrolled + num_impostor_identities


    X_basis, _, _ = generate_dataset(
        total_identities, samples_per_identity=20, seed=seed
    )
    model = EigenfaceModel(n_components=n_components).fit(X_basis)

   
    X_enroll, y_enroll, identities = generate_dataset(
        total_identities, samples_per_identity=enroll_samples, seed=seed + 1
    )
    X_query, y_query, _ = generate_dataset(
        total_identities, samples_per_identity=query_samples, seed=seed + 2
    )

    enrolled_ids = list(range(num_enrolled))
    impostor_ids = list(range(num_enrolled, total_identities))

    
    references = {}
    for ident in enrolled_ids:
        vecs = model.project_quantized(X_enroll[y_enroll == ident])
        references[ident] = np.round(vecs.mean(axis=0)).astype(np.int8)

    
    genuine_mask = np.isin(y_query, enrolled_ids)
    Xg, yg = X_query[genuine_mask], y_query[genuine_mask]
    Qg = model.project_quantized(Xg)

    genuine_min_dist = []
    genuine_predicted = []
    for vec, true_id in zip(Qg, yg):
        dists = {ident: euclidean_int8(vec, ref) for ident, ref in references.items()}
        best_id = min(dists, key=dists.get)
        genuine_min_dist.append(dists[best_id])
        genuine_predicted.append(best_id)
    genuine_min_dist = np.array(genuine_min_dist)
    genuine_predicted = np.array(genuine_predicted)
    genuine_correct_id = genuine_predicted == yg

    
    impostor_mask = np.isin(y_query, impostor_ids)
    Xi = X_query[impostor_mask]
    Qi = model.project_quantized(Xi)

    impostor_min_dist = []
    for vec in Qi:
        dists = [euclidean_int8(vec, ref) for ref in references.values()]
        impostor_min_dist.append(min(dists))
    impostor_min_dist = np.array(impostor_min_dist)

    return {
        "genuine_min_dist": genuine_min_dist,
        "genuine_correct_id": genuine_correct_id,
        "impostor_min_dist": impostor_min_dist,
        "references": references,
    }


def threshold_sweep(result, thresholds):
    rows = []
    gd, gcorrect = result["genuine_min_dist"], result["genuine_correct_id"]
    idist = result["impostor_min_dist"]

    for t in thresholds:
        accepted_genuine = gd <= t
       
        top1_correct = accepted_genuine & gcorrect
        frr = 1 - top1_correct.mean()
        far = (idist <= t).mean()
        rows.append((t, top1_correct.mean(), frr, far))
    return rows


if __name__ == "__main__":
    result = run_experiment()

    print("=== Distâncias (genuínas vs. impostoras) ===")
    print(f"Genuínas  -> min={result['genuine_min_dist'].min():.1f}  "
          f"max={result['genuine_min_dist'].max():.1f}  "
          f"média={result['genuine_min_dist'].mean():.1f}")
    print(f"Impostoras-> min={result['impostor_min_dist'].min():.1f}  "
          f"max={result['impostor_min_dist'].max():.1f}  "
          f"média={result['impostor_min_dist'].mean():.1f}")

    print(f"\nAcerto de identidade entre genuínas (ignorando threshold): "
          f"{result['genuine_correct_id'].mean() * 100:.1f}%")

    print("\n=== Varredura de threshold ===")
    print(f"{'thresh':>8} {'top1_acc':>10} {'FRR':>8} {'FAR':>8}")
    thresholds = np.arange(0, 260, 10)
    rows = threshold_sweep(result, thresholds)
    for t, acc, frr, far in rows:
        print(f"{t:8.0f} {acc*100:9.1f}% {frr*100:7.1f}% {far*100:7.1f}%")

    
    best = min(rows, key=lambda r: r[2] + r[3])
    print(f"\nMelhor ponto de operação: threshold={best[0]:.0f} "
          f"-> top1_acc={best[1]*100:.1f}%  FRR={best[2]*100:.1f}%  FAR={best[3]*100:.1f}%")
