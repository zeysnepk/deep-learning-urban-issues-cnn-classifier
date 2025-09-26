# Urban Issues Dataset CNN Sınıflandırma Projesi 🏙️

Bu proje, **Akbank Derin Öğrenme Bootcamp** kapsamında CNN (Convolutional Neural Network) mimarisi kullanarak kentsel sorunların görüntü sınıflandırması üzerine geliştirilmiştir.

## 📋 Proje Özeti

Urban Issues Dataset kullanılarak farklı kentsel problemlerin otomatik olarak tespit edilmesi ve sınıflandırılması amacıyla derin öğrenme modeli geliştirilmiştir. Proje, görüntü sınıflandırması, veri analizi, model geliştirme, değerlendirme ve yorumlama konularında kapsamlı bir yaklaşım sunmaktadır.

## 🎯 Proje Amacı

- Kentsel sorunları görüntü analizi ile otomatik tespit etmek
- CNN mimarisi ile yüksek doğrulukta sınıflandırma modeli geliştirmek
- Derin öğrenme tekniklerinde pratik deneyim kazanmak

## 📊 Veri Seti Hakkında

**Dataset**: Urban Issues Dataset (Kaggle)
- **Kaynak**: [Kaggle Urban Issues Dataset](https://www.kaggle.com/datasets/akinduhiman/urban-issues-dataset)
- **Veri Yapısı**: Çoklu sınıf görüntü sınıflandırma
- **Organizasyon**: Train/Validation/Test setlerine ayrılmış
- **Özel Yapı**: İç içe klasör yapısı (urban-issues-dataset/ClassName/ClassName/train-test-valid/images/)

### Veri Seti İstatistikleri
- **Toplam Sınıf Sayısı**: Değişken (dataset'e göre)
- **Görüntü Formatı**: RGB renkli görüntüler
- **Görüntü Boyutları**: Değişken boyutlarda (224x224'e yeniden boyutlandırıldı)
- **Veri Dağılımı**: Train/Validation/Test setlerine dengeli dağıtım

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler

### Deep Learning Framework
- **TensorFlow 2.x**: Ana derin öğrenme framework'ü
- **Keras**: High-level neural network API

### Veri İşleme ve Görselleştirme
```python
- NumPy: Numerical computing
- Pandas: Data manipulation
- Matplotlib & Seaborn: Data visualization
- OpenCV: Image processing
- Scikit-learn: Machine learning utilities
```

### Özel Kütüphaneler
- **ImageDataGenerator**: Data augmentation

## 🔧 Kullanılan Yöntemler

### 1. Veri Ön İşleme
- **Görüntü Normalizasyonu**: Pixel değerlerini [0,1] aralığına normalize etme
- **Boyut Standardizasyonu**: Tüm görüntüleri 224x224 boyutuna getirme
- **Veri Yapısı Düzenleme**: İç içe klasör yapısını düzeltme

### 2. Data Augmentation Teknikleri
```python
- Rotation: ±20° döndürme
- Width/Height Shift: %20 kaydırma  
- Horizontal Flip: Yatay çevirme
- Zoom: %20 zoom
- Shear: %15 yamultma
- Brightness: Parlaklık değişimi
```

### 3. Model Mimarileri

#### Custom CNN Modeli
- **Konvolüsyon Blokları**: 4 adet Conv2D bloğu
- **Filtre Sayıları**: 32 → 64 → 128 → 256
- **Regularization**: Dropout (0.25-0.5) + BatchNormalization
- **Pooling**: MaxPooling2D (2x2)
- **Dense Katmanlar**: 512 → 256 → num_classes
- **Aktivasyon**: ReLU (hidden), Softmax (output)

#### Transfer Learning Modeli
- **Base Model**: VGG16 (ImageNet pre-trained)
- **Fine-tuning**: Son 5 katman eğitilebilir
- **Custom Layers**: GlobalAveragePooling2D + Dense layers
- **Regularization**: Dropout + BatchNormalization

### 4. Optimizasyon Teknikleri
- **Optimizer**: Adam optimizer
- **Learning Rate**: 0.001 (Custom), 0.0001 (Transfer)
- **Loss Function**: Categorical Crossentropy
- **Class Weights**: Dengesiz veri için ağırlıklandırma
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### 5. Model Değerlendirme Metrikleri
- **Accuracy**: Genel doğruluk oranı
- **Precision**: Sınıf bazında kesinlik
- **Recall**: Sınıf bazında duyarlılık  
- **F1-Score**: Precision ve Recall'ın harmonik ortalaması
- **Confusion Matrix**: Sınıf karışıklık matrisi

### 6. Model Açıklanabilirliği
- **Feature Visualization**: Özellik haritalarının görselleştirilmesi
- **Misclassified Analysis**: Yanlış sınıflandırılan örneklerin analizi

## 📈 Elde Edilen Sonuçlar

### Sınıf Bazında Performans
- **En İyi Performans**: Damaged concrete structures (F1: 0.9698)
- **En Kötü Performans**: IllegalParking (F1: 0.1695)

### Önemli Bulgular
- Model overfitting göstermedi (regularization teknikleri etkili)

## 🚀 Projeyi Çalıştırma

### Kaggle'da Çalıştırma (Önerilen)
1. **Kaggle hesabınızla giriş yapın**
2. **Urban Issues Dataset'i ekleyin**
3. **Yeni notebook oluşturun**
4. **Kod hücrelerini sırayla çalıştırın**

### Yerel Ortamda Çalıştırma
```bash
# Repository'yi klonlayın
git clone https://github.com/zeysnepk/deep-learning-urban-issues-cnn-classifier.git
cd deep-learning-urban-issues-cnn-classifier

# Gerekli paketleri yükleyin
pip install -r requirements.txt

# Jupyter notebook'u başlatın
jupyter notebook
```

### Gerekli Paketler (requirements.txt)
```txt
tensorflow>=2.8.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
opencv-python>=4.5.0
Pillow>=8.3.0
```

## 🔍 Detaylı Analiz ve Sonuçlar

### Eğitim Süreci Analizi
- **Overfitting Kontrolü**: Validation loss eğrisi ile monitör edildi
- **Learning Rate Scheduling**: ReduceLROnPlateau ile otomatik ayarlama
- **Early Stopping**: 10 epoch patience ile erken durdurma

### Hata Analizi
- **Yanlış Sınıflandırma Oranı**: %16.68
- **Karışan Sınıflar**: Benzer kentsel problemler arası karışıklık
- **İyileştirme Önerileri**: Daha fazla veri ve ensemble yöntemleri

## 🎯 Gelecek İyileştirmeler

### Kısa Vadeli İyileştirmeler
- [ ] **Ensemble Methods**: Birden fazla modeli birleştirme
- [ ] **Advanced Augmentation**: Mixup, CutMix teknikleri
- [ ] **Hyperparameter Optimization**: Bayesian optimization
- [ ] **Model Compression**: Quantization ve pruning

### Uzun Vadeli Geliştirmeler
- [ ] **Object Detection**: YOLO/R-CNN ile nesne tespiti
- [ ] **Semantic Segmentation**: Pixel-level sınıflandırma
- [ ] **Multi-modal Learning**: Metin + görüntü kombinasyonu
- [ ] **Real-time Deployment**: Mobile/edge deployment

## 👨‍💻 Geliştirici

**Zeynep K.**
- 🐙 GitHub: [@zeysnepk](https://github.com/zeysnepk)
- 💼 LinkedIn: [linkedin-profil](https://www.linkedin.com/in/-zeynepkaplan)

## 🏆 Bootcamp Bilgileri

- **Program**: Akbank Derin Öğrenme Bootcamp
- **Tarih**: Eylül 2025
- **Mentor**: Defne Buse Çelik

## 📄 Lisans

Bu proje MIT lisansı altında açık kaynak olarak paylaşılmıştır. Detaylar için `LICENSE` dosyasına bakınız.

## 🙏 Teşekkürler

- **Akbank** ve **Bootcamp Organizatörleri** eğitim fırsatı için
- **Kaggle Community** veri seti ve notebook örnekleri için
- **TensorFlow/Keras Team** harika framework için
- **Açık Kaynak Topluluğu** kullanılan tüm kütüphaneler için
