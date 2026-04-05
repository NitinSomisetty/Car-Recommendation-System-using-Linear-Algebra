# Vehicle Recommendation Engine using QR Decomposition, Eigenanalysis and Least Squares Approximation

A Linear Algebra based car recommendation system built on the CarDekho dataset (~15,000 entries). Given a user's preferences — budget, mileage, fuel type, transmission, and more — the system recommends the top 5 most similar cars using matrix operations and vector projection.

---

## Methods Used

| Stage                 | Method                                         | Purpose                                                     |
| --------------------- | ---------------------------------------------- | ----------------------------------------------------------- |
| Matrix Representation | One-hot encoding, feature matrix construction  | Convert raw CSV data into a mathematical matrix             |
| Matrix Simplification | RREF via SymPy                                 | Identify linearly independent features                      |
| Structure of Space    | Rank & Nullity (NumPy)                         | Confirm feature independence                                |
| Remove Redundancy     | Linear independence test, column dropping      | Eliminate dummy variable trap and redundant features        |
| Orthogonalization     | QR Decomposition (Gram-Schmidt)                | Decorrelate features for meaningful distance computation    |
| Projection            | Euclidean distance in orthogonal feature space | Find closest matching cars to user preference vector        |
| Least Squares         | `numpy.linalg.lstsq`                           | Best approximation of user vector in car feature space      |
| Pattern Discovery     | Eigenvalue decomposition of covariance matrix  | Reveal which components drive the most variance across cars |
| System Simplification | Diagonalization                                | Decompose covariance matrix into independent components     |

---

## Concepts Learned

- Representing real-world data as a matrix and understanding what each entry means physically
- RREF and rank to detect redundant features — the dummy variable trap
- Rank-Nullity theorem: rank + nullity = total columns
- Gram-Schmidt orthogonalization and QR decomposition — why orthogonal bases matter for distance calculations
- Feature scaling (StandardScaler) — why unscaled features break distance metrics
- Least squares as the best approximation of a vector not in the column space
- Eigenvalues as variance explained — why high eigenvalue = important component
- Diagonalization: A = PDP⁻¹ and what it reveals about data structure

---

## How to Run

### Requirements

```
pip install pandas numpy sympy scikit-learn matplotlib
```

### Steps

1. Download the CarDekho dataset from Kaggle and place `cardekho_dataset.csv` in the same directory as `main.ipynb`
2. Open `main.ipynb` in Jupyter Notebook or VS Code
3. Run all cells top to bottom (Kernel → Restart & Run All)
4. When prompted, enter your car preferences:
   - Vehicle age, km driven, mileage, engine CC, max power, seats, budget
   - Seller type: `Dealer` or `Individual`
   - Fuel type: `Diesel` or `Petrol`
   - Transmission: `Automatic` or `Manual`
5. Top 5 recommended cars will be printed at the end

---

## Dataset

- **Source:** CarDekho Dataset — [Kaggle](https://www.kaggle.com/)
- **Size:** ~15,000 entries
- **Features used:** vehicle_age, km_driven, mileage, engine, max_power, seats, selling_price, seller_type, fuel_type, transmission_type

---

## Future Scaling

- **SVD-based recommendation:** Replace distance metric with Singular Value Decomposition for a proper matrix factorization recommender — the same approach used by Netflix and Spotify
- **PCA visualization:** Plot cars in 2D using top 2 eigenvectors to visually show clusters by price range or fuel type
- **Streamlit web app:** Deploy as an interactive web application so users don't need to run a notebook
- **Richer dataset:** Combine with car review sentiment data for a multi-source recommendation system
- **Collaborative filtering:** Incorporate user rating history for personalized recommendations beyond feature similarity
