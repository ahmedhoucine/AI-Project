# Verify no NaNs
assert not np.isnan(data_prepared.data).any(), "NaNs still present in data"

# Prepare target
y = data['Job Title'].values

# Split data maintaining sparse format
train_indices, test_indices = train_test_split(
    np.arange(len(y)), test_size=0.2, random_state=42
)
X_train = data_prepared[train_indices]
y_train = y[train_indices]
X_test = data_prepared[test_indices]
y_test = y[test_indices]

# Reduce dimensionality for better performance
svd = TruncatedSVD(n_components=500)
X_train_reduced = svd.fit_transform(X_train)
X_test_reduced = svd.transform(X_test)

# Models
models = {
    'Logistic Regression': LogisticRegression(
        solver='liblinear',
        penalty='l2',
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    ),
    'Linear SVM': LinearSVC(
        penalty='l2',
        loss='squared_hinge',
        dual=True,
        random_state=42,
        class_weight='balanced'
    ),
    'SGD Classifier': SGDClassifier(
        loss='log_loss',
        penalty='l2',
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
}

# Train and evaluate
for name, model in models.items():
    print(f"\n=== Training {name} ===")
    model.fit(X_train_reduced, y_train)
    y_pred = model.predict(X_test_reduced)
    
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    print("\nSample predictions:")
    print("Actual:", y_test[:5])
    print("Predicted:", y_pred[:5])

# Feature importance (for logistic regression)
if 'Logistic Regression' in models:
    lr = models['Logistic Regression']
    importance = np.abs(lr.coef_[0])
    top_features = importance.argsort()[-10:][::-1]
    print("\nTop 10 important features:")
    print(top_features)
    print("Importance scores:", importance[top_features])