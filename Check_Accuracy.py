# สมมติว่ามีผลลัพธ์การทดสอบที่เป็นจริง (actual) และที่โมเดลทำนาย (predicted)
# actual = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
# predicted = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0]

actual = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
predicted = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]

# คำนวณค่า Accuracy
correct_predictions = sum(a == p for a, p in zip(actual, predicted))
total_predictions = len(actual)
accuracy = correct_predictions / total_predictions * 100

# คำนวณค่า Error Rate
error_rate = (total_predictions - correct_predictions) / total_predictions * 100

print(f"Accuracy: {accuracy:.2f}%")
print(f"Error Rate: {error_rate:.2f}%")
