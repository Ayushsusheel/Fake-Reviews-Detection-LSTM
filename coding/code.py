Algorithm :

1. LSTM-Based Fake Review Classification
 

1. procedure LSTMClassifier
2. D ← LoadDataset()
3. Dlabel ← MapLabels(FR → 1, OR → 0)
4. Dtext ← CleanText(D)
5. T ← Tokenizer(V = 5000)
6. X ← Pad(T (Dtext), len = 100)
7. y ← Dlabel
8. (Xtrain, Xtest, ytrain, ytest) ← Split(X, y, 80:20)
9. Model ← Embedding → Dropout → LSTM → FC → σ
10. L ← BinaryCrossEntropy()
11. O ← Adam(η = 10−3, λ = 10−4)
12. for e = 1 to E do
13.      TrainLoss, TrainAcc ← 0
14.      for all (xi, yi) ∈ TrainLoader do
15.          y ̂i ← Model(xi)
16. 	ℓ ← L (y ̂i, yi)
17. 	O.step(∇ℓ)
18. 	Update TrainLoss, TrainAcc
19.      end for
20.      Evaluate on validation set: ValLoss, ValAcc
21.      if ValLoss < BestLoss then
22. 	Save model
23. 	Reset patience counter
24.      else
25. 	Increment patience counter
26. 	if patience limit reached then
27. 	   break
28. 	end if
29.      end if
30. end for
31. Y ̂ ← Model(Xtest)
32. P ← Threshold(Y ̂, 0.5)
33. Evaluate metrics: Accuracy, Precision, Recall, F1
34. Visualize: Confusion Matrix, ROC, Training/Loss curves
35. end procedure
