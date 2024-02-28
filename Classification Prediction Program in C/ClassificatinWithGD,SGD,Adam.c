#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define DICTIONARY_SIZE 600
#define TRAIN_PERCENTAGE 80
#define LEARNING_RATE 0.1
#define EPOCHS 100
#define EPSILON 1e-8
#define BETA1 0.9
#define BETA2 0.999

// Fonksiyon bildirimleri
void dosyaOkuma(char ***sozluk,char *dosyaadi, int label, int *asilIndisler, int *kelimeSayisi, int *cumleSayisi, int *,int **,int *);
void convertTohotVector(char sentence[]);
int findWordIndex(char word[], int *kelimeSayisi, char ***sozluk);
void gradientDescent(int **trainData, int *trainLabels, int trainSize, int cumleSayisi, int epochs, double learningRate, double *W, int kelimeSayisi,int **hotVector, double ***jacobian,double ***tJacobian, int **testData, int *testLabels);
void stochasticGradientDescent(int **trainData, int *trainLabels, int trainDataSize, int cumleSayisi, int epochs, double learningRate, double *W, int kelimeSayisi,int **hotVector, double ***jacobian,double ***tJacobian, int**testData, int *testLabels);
void hotVectoreAta(char ***sozluk, int ***hotVector, int *kelimeSayisi, int *cumleSayisi, int *asilIndisler);
void adam(int **trainData, int *trainLabels, int trainDataSize, int cumleSayisi, int epochs, double learningRate, double *W, int kelimeSayisi,int **hotVector, double ***jacobian,double ***tJacobian, int**testData, int *testLabels,double **m,double **v, double **m_yeni, double **v_yeni);
int oncekilerleKiyas(int *indices, int anlikIndeks, int kiyasSayi);


int main() {
	srand(time(NULL));
	int i,j,k=0,l=0;
	int **hotVector;
	char **sozluk;
	double *W;
	int *labels;
	sozluk = NULL;
	labels = NULL;
	double **jacobian;
	jacobian = NULL;
	double **tJacobian;
	tJacobian = NULL;
	double *r;
	r = NULL;
	double *m;
	m=NULL;
	double *v;
	v=NULL;
	double *m_yeni;
	m_yeni = NULL;
	double *v_yeni;
	v_yeni = NULL;
    char *dosyaadi1 = "musto.txt";
    char *dosyaadi2 = "taha.txt";
	int *asilIndisler;
	int sonIndis = 0;
	int kelimeSayisi = 0;
	int cumleSayisi = 0;
	int ilkDosyaninCumleSayisi = 0;
    // A sinifi (label = 1) verileri oku
	asilIndisler = calloc(250 * 25, sizeof(int));
    dosyaOkuma(&sozluk,dosyaadi1, 1, asilIndisler, &kelimeSayisi, &cumleSayisi, &sonIndis,&labels,&ilkDosyaninCumleSayisi);
    // B sinifi (label = -1) verileri oku
    dosyaOkuma(&sozluk,dosyaadi2, -1, asilIndisler, &kelimeSayisi, &cumleSayisi, &sonIndis,&labels,&ilkDosyaninCumleSayisi);
	hotVectoreAta(&sozluk,&hotVector, &kelimeSayisi, &cumleSayisi, asilIndisler);
	
  	W = (double *)malloc(kelimeSayisi * sizeof(double));
  	//Eðitilecek dizi oluþturuldu
  	
    // Egitim ve test kümesine bölme
    int totalDataSize = cumleSayisi;
    int trainSize = (TRAIN_PERCENTAGE * totalDataSize) / 100;
    //Eðitim kümesi toplam kümenin %80i alýndý

    // Egitim kümesi
    int **trainData = (int **)malloc(trainSize * sizeof(int *));
    int *trainLabels = (int *)malloc(trainSize * sizeof(int));

    // Test kümesi
    int **testData = (int **)malloc((totalDataSize - trainSize) * sizeof(int *));
    int *testLabels = (int *)malloc((totalDataSize - trainSize) * sizeof(int));

    // Rastgele siralama için indeks dizisi olustur
    int *indices = (int*)malloc(totalDataSize * sizeof(int));
    for (i = 0; i < totalDataSize; i++) {
        indices[i] = i;
    }

    // Indeksleri karistir
    for ( i = 0; i < totalDataSize - 1; i++) {
        int j = rand() % (totalDataSize);
        while(oncekilerleKiyas(indices,i,j)){
        j = rand() % (i + 1);}
        indices[i] = j;
    }
    
    int temp = indices[totalDataSize-1];
    indices[totalDataSize-1] = indices[0];
    indices[0] = temp;

    // Egitim ve test kümesine rastgele verileri ata
    for ( i = 0; i < totalDataSize; i++) {
        int *data = malloc(cumleSayisi * sizeof(double));
        data = hotVector[indices[i]];
        
        if (i < trainSize) {
            trainData[i] = data;
            trainLabels[i] = labels[indices[i]];  // A sinifi için etiket 1, B sinifi için etiket -1
        } 
		else {
            testData[i - trainSize] = data;
            testLabels[i - trainSize] = labels[indices[i]];
        }
    }
    //gradientDescent(trainData, trainLabels, trainSize, cumleSayisi, EPOCHS, LEARNING_RATE, W, kelimeSayisi,hotVector,&jacobian,&tJacobian,testData,testLabels);
    stochasticGradientDescent(trainData, trainLabels, trainSize, cumleSayisi, EPOCHS, LEARNING_RATE, W, kelimeSayisi,hotVector,&jacobian,&tJacobian,testData,testLabels);
	//adam(trainData, trainLabels, trainSize, cumleSayisi, EPOCHS, LEARNING_RATE, W, kelimeSayisi,hotVector,&jacobian,&tJacobian,testData,testLabels,&m,&v,&m_yeni,&v_yeni);
	
	//Bellej serbest býrakma
    free(trainData);
    free(trainLabels);
    free(testData);
    free(testLabels);
    free(indices);

    return 0;
}

// Dosyadan veri okuma fonksiyonu
void dosyaOkuma(char ***sozluk, char *dosyaadi, int label, int *asilIndisler, int *kelimeSayisi, int *cumleSayisi, int *sonIndis, int **labels, int *ilkDosyaninCumleSayisi) {
    FILE *pf;
    char buffer[100];
    int i, j;

    pf = fopen(dosyaadi, "r");
    //dosya açma
    if (pf == NULL) {
        printf("Unable to open the file: %s\n", dosyaadi);
        //dosya yoksa hata verdik
    } else {
        while (fgets(buffer, sizeof(buffer), pf) != NULL) {
            (*cumleSayisi)++;
            //Dosya açtýktan sonra dosyalarý satýr satýr okuduk
            const char *ayraclar = " ,!\"#$%&'()+./:;<=>?@[\\]^_`{|}~";
            char *kelime = strtok(buffer, ayraclar);
            int k = 0;
            //Satýrlarda noktalama iþaretlerini bir veri almasýn diye onlarý sildik.
            if (strlen(kelime) > 0 && kelime[strlen(kelime) - 1] == '\n') {
                kelime[strlen(kelime) - 1] = '\0';
            }
            //sildikten sonra sonlarýna null deðerini atadýk
            while (kelime != NULL) {
                k = 0;

                if ((*sozluk) != NULL) {
                    for (j = 0; j < strlen(kelime); j++) {
                        kelime[j] = tolower(kelime[j]);
                    }
                    //kelimeleri okuduktan sonra büyük harf ve küçük harfle yazýlan ayný kelimeleri farklý veri almamasý için tüm harflerini küçüðe çevirdik
                    while (k < *kelimeSayisi && strcmp((*sozluk)[k], kelime) != 0) {
                        k++;
                    }
                }
                //sözlükte farklý kelimeleri tutuyoruz. Ayýrdýðýmýz kelimenin sözlükte aynýsý var mý kontrolü yapýyoruz 
                if (k == *kelimeSayisi && *kelime != '\n') {
                    *sozluk = (char **)realloc(*sozluk, (*kelimeSayisi + 1) * sizeof(char *));
                    (*sozluk)[*kelimeSayisi] = strdup(kelime);
                    (*kelimeSayisi)++;
                }
                //Kelime farklýysa sözlüge atýyoruz ve sözlük boyutunu geniþletiyoruz.
                if (*kelime != '\n') {
                    int indis = findWordIndex(kelime, kelimeSayisi, sozluk);
                    asilIndisler[(*sonIndis)++] = indis;
                }
                //Ýlk satýrda cümlemiz bittiyse cümlenin kelime olarak kaçýncý kelimede bittiðini belirtmek için asilIndisler dizisi tanýmladýk.
                //Bu dizi bir cümlenin kelimelerini sözlükteki indis þeklinde gösteriyor aslýnda.
                //Bu dizide farklý kelime sayýlarýn sözlükteki indisleri tutuluyor ve -1 olan yerlerinde aldýðýmýz cümle bitiyor.
                kelime = strtok(NULL, ayraclar);
            }
            //Burada cümle bittiði için sonuna -1 atadýk.
            asilIndisler[(*sonIndis)++] = -1;
        }

        (*labels) = (int *)realloc((*labels), (*cumleSayisi + 1) * sizeof(int));
        
        for (i = (*ilkDosyaninCumleSayisi); i < (*cumleSayisi); i++) {
            (*labels)[i] = label;
        }
		//Etiket dizisi oluþturduk.Ýlk dosyadan okuduðumuz bir konu ile alakalý verilere 1, diðer dosyadan okuduklarýmýza -1 yapýþtýracaðýz
        (*ilkDosyaninCumleSayisi) = (*cumleSayisi);
        //ilk dosyanýn kaldýðý yerinden devam edebilmek için böyle bir iþlem yaptýk
        fclose(pf);
        //dosya kapama
    }
}

//Bu fonksiyonda aradýðýmýz kelimenin sözlükteki indisini döndürdük     
int findWordIndex(char word[], int *kelimeSayisi, char ***sozluk) {
	int i;
    for ( i = 0; i < *kelimeSayisi; i++) {
        if ((*sozluk)[i] != NULL && strcmp((*sozluk)[i], word) == 0) {
            return i;
        }
    }
}

//Hot Vector oluþturma
void hotVectoreAta(char ***sozluk, int ***hotVector, int *kelimeSayisi, int *cumleSayisi, int *asilIndisler){
  int anlikIndis=0;
  int i,j;
  (*hotVector) = (int **)malloc((*cumleSayisi) * sizeof(int *));
  for(i=0;i<(*cumleSayisi);i++){
    (*hotVector)[i]=(int*)malloc((*kelimeSayisi)*sizeof(int));
  }
	//hotVector dizisi oluþturduk
  for(i=0;i<(*cumleSayisi);i++){
  	for(j=0; j< (*kelimeSayisi) ; j++){
    	(*hotVector)[i][j] = 0;
    }
    //ilk deðerlerini 0 atadýk.
    while( asilIndisler[anlikIndis] != -1){
      	(*hotVector)[i][asilIndisler[anlikIndis]] = 1;
      	anlikIndis++;
    }
    anlikIndis++;
    //hotvectoru asilIndislerde -1 görene kadar yani cümle sonunu görene kadar döndürdük ve oradaki indisine sözlükteki kelime varsa 1 yoksa 0 atadýk.
  }	
}

//Gradient Descent
void gradientDescent(int **trainData, int *trainLabels, int trainDataSize, int cumleSayisi, int epochs, double learningRate, double *W, int kelimeSayisi, int **hotVector, double ***jacobian, double ***tJacobian, int **testData, int *testLabels) {
    int i, j, epoch,k;
    double matrisCarpimi = 0;
    double matrisCarpimlari[200];
    double mse;
    double prediction;
    double random;
	double jtoplami;
	clock_t start, end;
    double cpu_time_used;
	FILE *fptr = fopen("loss.txt", "a"); 
	FILE *fptr2 = fopen("time.txt", "a");
	FILE *fptr3 = fopen("epochs.txt", "a");
	FILE *fptr4 = fopen("W.txt", "w");
    (*jacobian) = (double **)malloc(cumleSayisi * sizeof(double *));
    for (i = 0; i < cumleSayisi; i++) {
        (*jacobian)[i] = (double *)malloc(kelimeSayisi * sizeof(double));
    }
    //jocbian matrisi için yer ayýrdýk

    srand(time(NULL));

    for (i = 0; i < kelimeSayisi; i++) {
        random = sin(rand() * rand());
        random = -1 + (1 - (-1)) * fabs(random);
        W[i] = random;
    }
    //Eðitmemiz gereken W dizisinin tüm elemanlarýna rastgele deðer atadýk
    for (epoch = 0; epoch < epochs; epoch++) {   // iterasyon sayýsý kadar eðitiyoruz
    start = clock();
        for (i = 0; i < trainDataSize; i++) {
            matrisCarpimi = 0;

            for (j = 0; j < kelimeSayisi; j++) {
                matrisCarpimi += W[j] * ((double)(trainData[i][j]));
            }
            //rastgele deðerlerimizi eðitim kümemizdeki hotVectorler ile çarpýyoruz
            matrisCarpimlari[i] = trainLabels[i] - matrisCarpimi;
            //Eðitim kümemizin etiketlerinden bulduðumuz deðeri çýkarýyoruz
        }

        for (i = 0; i < trainDataSize; i++) {
            for (j = 0; j < kelimeSayisi; j++) {
                (*jacobian)[i][j] = -1 * ((double)(trainData[i][j])) * (pow(cosh(matrisCarpimlari[i]), -2));
                //gradiant vectorumuz olmadýðý için nonlineer regression yapýyoruz.Eðitim kümemizden tanh türevini alarak jacobian matrisi oluþturuyoruz
            }
        }

        for (i = 0; i < kelimeSayisi; i++) {
            jtoplami = 0;

            for (j = 0; j < trainDataSize; j++) {
                jtoplami += matrisCarpimlari[j] * (*jacobian)[j][i];
            }
            //jacobian transpozesini aldýk ve ilk eðitim kümemizi eðitmeye baþladýðýmýz matrisle çarpýyoruz ve çýkan deðerleri topluyoruz
            W[i] = W[i] - learningRate * jtoplami;
			for (i = 0; i < kelimeSayisi; i++)
		fprintf(fptr4,"%f ", W[i]);
		fprintf(fptr4,"\n");
            //Burada GD uyguluyoruz jtoplami deðerimizi dizi olarak da tutabilirdik döngüne tutmak istemedik jtoplami=gradiantvector
        }
        

        mse=0;
        for (i = 0; i < cumleSayisi - trainDataSize; i++) {
            double prediction = 0;

            for (j = 0; j < cumleSayisi; j++) {
                prediction += W[j] * (double)testData[i][j];
            }

            prediction = tanh(prediction);            
            mse=testLabels[i]-prediction;
            mse+=mse*mse;            
            //if(epoch==1)
            //printf("%d . cumlenin egitilmis W's = %f, predictionu : %f , gercek labeli : %d  loss=%f \n",i,W[i],prediction,testLabels[i],mse);
        }
        mse=mse/(cumleSayisi - trainDataSize);
        //printf("mse=%f\n",mse);
        fprintf(fptr,"%f\n", mse);
        //test kümemizle eðitilmiþ W vektörümüzü test ediyoruz
        end = clock();
     	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
     	fprintf(fptr2,"%f\n",cpu_time_used );
    	//printf("Gecen sure: %f saniye\n", cpu_time_used);
    	fprintf(fptr3,"%d\n",epoch );
    }
    fclose(fptr);
    fclose(fptr2);
    fclose(fptr3);
    fclose(fptr4);
}

//SGD iþlemi
void stochasticGradientDescent(int **trainData, int *trainLabels, int trainDataSize, int cumleSayisi, int epochs, double learningRate, double *W, int kelimeSayisi, int **hotVector, double ***jacobian, double ***tJacobian, int **testData, int *testLabels) {
    int i, j, epoch;
    double matrisCarpimi = 0;
    double matrisCarpimlari[200];
    double random;
    double toplam=0;
	double jtoplami = 0;
	double prediction;
	double mse;
	clock_t start, end;
    double cpu_time_used;
    FILE *fptr = fopen("loss.txt", "a"); 
	FILE *fptr2 = fopen("time.txt", "a");
	FILE *fptr3 = fopen("epochs.txt", "a");
	FILE *fptr4 = fopen("W.txt", "w");
    
	
    (*jacobian) = (double **)malloc(cumleSayisi * sizeof(double *));
    for (i = 0; i < cumleSayisi; i++) {
        (*jacobian)[i] = (double *)malloc(kelimeSayisi * sizeof(double));
    }
    //Jacobian matrisi için yer ayýrdýk
    srand(time(NULL));

    for (i = 0; i < kelimeSayisi; i++) {
        random = sin(rand() * rand());
        random = -1 + (1 - (-1)) * fabs(random);
        W[i] = random;
    }
    //Eðitmemiz gereken W dizisinin tüm elemanlarýna rastgele deðer atadýk
    for (epoch = 0; epoch < trainDataSize; epoch++) {   //eðitim kümesinden sýrayla verileri seçtik ve eðitim kümesi kadar eðitiyoruz
        matrisCarpimi = 0;
        start = clock();
        for (j = 0; j < kelimeSayisi; j++) {
            matrisCarpimi += W[j] * ((double)(trainData[epoch][j]));
        }
        //test kümemle random W kümemi eðitmeye baþladým
        matrisCarpimlari[epoch] = trainLabels[epoch] - matrisCarpimi;
        //etiketinden hesaba devam ediyorum
        for (j = 0; j < kelimeSayisi; j++) {
            (*jacobian)[epoch][j] = -1 * ((double)(trainData[epoch][j])) * (pow(cosh(matrisCarpimlari[epoch]), -2));
        }
        //gradiant vectorumuz olmadýðý için nonlineer regression yapýyoruz.Eðitim kümemizden tanh türevini alarak jacobian matrisi oluþturuyoruz
        
        for (i = 0; i < kelimeSayisi; i++) {
        	jtoplami=0;
            jtoplami = matrisCarpimlari[epoch] * (*jacobian)[epoch][i];
			W[i] = W[i] - learningRate * jtoplami; 
			for (j = 0; j < kelimeSayisi; j++)
		fprintf(fptr4,"%f ", W[j]);
		fprintf(fptr4,"\n");                     
        }    
		mse=0;    
        //jacobian transpozesini aldýk ve ilk eðitim kümemizi eðitmeye baþladýðýmýz matrisle çarpýyoruz ve çýkan deðerleri topluyoruz
    	for (i = 0; i < cumleSayisi - trainDataSize; i++) {
    		prediction = 0;
    		for (j = 0; j < kelimeSayisi; j++) {
				prediction += W[i] * (double)testData[i][j];
			}
			prediction = tanh(prediction);
			mse=testLabels[i]-prediction;
            mse+=mse*mse; 
			//if(epoch==0||epoch==cumleSayisi-trainDataSize-1)
			//printf("%d . cumlenin egitilmis W's = %f, predictionu : %f , gercek labeli : %d \n",i,W[i],prediction,testLabels[i]);						                                        
    }    
    mse=mse/(cumleSayisi - trainDataSize);
    //printf("mse=%f\n",mse);
    fprintf(fptr,"%f\n", mse);
    //test kümemizle eðitilmiþ W vektörümüzü test ediyoruz
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    fprintf(fptr2,"%f\n",cpu_time_used );
    //printf("Gecen sure: %f saniye\n", cpu_time_used);
    fprintf(fptr3,"%d\n",epoch );
}
	fclose(fptr);
    fclose(fptr2);
    fclose(fptr3);
    fclose(fptr4);
}
//adam algoritmasý
void adam(int **trainData, int *trainLabels, int trainDataSize, int cumleSayisi, int epochs, double learningRate, double *W, int kelimeSayisi, int **hotVector, double ***jacobian, double ***tJacobian, int **testData, int *testLabels, double **m, double **v, double **m_yeni, double **v_yeni) {
    double *gradients;
    double mse;
    int i, j, k;
    clock_t start, end;
    double cpu_time_used;
    FILE *fptr = fopen("loss.txt", "a"); 
	FILE *fptr2 = fopen("time.txt", "a");
	FILE *fptr3 = fopen("epochs.txt", "a");
	FILE *fptr4 = fopen("W.txt", "a");
    *m = (double *)malloc(kelimeSayisi * sizeof(double));
    //1.moment vector için yer ayýrdýk
    *v = (double *)malloc(kelimeSayisi * sizeof(double));
    //2.moment vectoru için yer ayýrdýk
    *m_yeni = (double *)malloc(kelimeSayisi * sizeof(double));
    ////1.moment vector güncellemesi için yer ayýrdýk
    *v_yeni = (double *)malloc(kelimeSayisi * sizeof(double));
    //2.moment vector güncellemesi için yer ayýrdýk
    gradients = (double *)malloc(kelimeSayisi * sizeof(double));
    //gradient vector için yer ayýrdýk

    (*jacobian) = (double **)malloc(cumleSayisi * sizeof(double *));
    for (i = 0; i < cumleSayisi; i++) {
        (*jacobian)[i] = (double *)malloc(kelimeSayisi * sizeof(double));
    }
    //jacobian matrisi için yer ayýrdýk

    double random;
    double matrisCarpimi = 0;
    double matrisCarpimlari[200];
    int epoch;
    double jtoplami;
    srand(time(NULL));

    for (i = 0; i < kelimeSayisi; i++) {
        random = sin(rand() * rand());
        random = -1 + (1 - (-1)) * fabs(random);
        W[i] = random;
    }
    //eðiteceðimiz diziye baþlangýçta rastgele deðerler atýyoruz

    for (i = 0; i < trainDataSize; i++) {
        matrisCarpimi = 0;
        for (j = 0; j < kelimeSayisi; j++) {
            matrisCarpimi += W[j] * ((double)(trainData[i][j]));
        }
        matrisCarpimlari[i] = trainLabels[i] - matrisCarpimi;
    }
    //Eðiteceðimizz diziyi eðitmeye baþlýyoruz

    for (i = 0; i < trainDataSize; i++) {
        for (j = 0; j < kelimeSayisi; j++) {
            (*jacobian)[i][j] = -1 * ((double)(trainData[i][j])) * (pow(cosh(matrisCarpimlari[i]), -2));
        }
    }
    //gradiant vectorumuz olmadýðý için nonlineer regression yapýyoruz.Eðitim kümemizden tanh türevini alarak jacobian matrisi oluþturuyoruz
    
    
    for (i = 0; i < kelimeSayisi; i++) {
        jtoplami = 0;
        for (j = 0; j < trainDataSize; j++) {
            jtoplami += matrisCarpimlari[j] * (*jacobian)[j][i];
        }
        gradients[i] = jtoplami;
    }
    //gradients jacobian transpozesiyle vectorumuzu hesapladýk

    for (epoch = 1; epoch < epochs; epoch++) {
    	start = clock();
        for (i = 0; i < kelimeSayisi; i++) {
            (*m)[i] = BETA1 * (*m)[i] + (1 - BETA1) * gradients[i];
            (*v)[i] = BETA2 * (*v)[i] + (1 - BETA2) * (gradients[i] * gradients[i]);
        }
        // ilk ve ikinci moment vectorlerimizi güncelledik

        for (i = 0; i < kelimeSayisi; i++) {
            (*m_yeni)[i] = (*m)[i] / (1 - pow(BETA1, epoch));
            (*v_yeni)[i] = (*v)[i] / (1 - pow(BETA2, epoch));
        }
        //ilk ve ikinci moment vectorlerimizi yeni deðerleri ile güncelledik

        for (i = 0; i < kelimeSayisi; i++) {
            W[i] = W[i] - ((double)(learningRate * (*m_yeni)[i] / (sqrt(((*v_yeni)[i])) + pow(10, -8))));
            for (j = 0; j < kelimeSayisi; j++)
		fprintf(fptr4,"%f ", W[j]);
		fprintf(fptr4,"\n");      
                //Sýfýra bölünme hatasý almamak için epsilon deðerimizi ekledik
        }
        //W eðitildi
        mse=0;
        for (i = 0; i < cumleSayisi - trainDataSize; i++) {
            double prediction = 0;
            for (j = 0; j < kelimeSayisi; j++) {
                prediction += W[j] * (double)testData[i][j];
            }
            prediction = tanh(prediction);
            mse=testLabels[i]-prediction;
            mse+=mse*mse;
            //if(epoch==1 || epoch==epochs-1)
            //printf("%d . cumlenin egitilmis W's = %f, predictionu : %f , gercek labeli : %d \n",i,W[i],prediction,testLabels[i]);
        }
        mse=mse/(cumleSayisi - trainDataSize);
    	//printf("mse=%f\n",mse);
    	fprintf(fptr,"%f\n", mse);
    	//test kümemizle eðitilmiþ W vektörümüzü test ediyoruz
    	end = clock();
    	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    	fprintf(fptr2,"%f\n",cpu_time_used );
    	//printf("Gecen sure: %f saniye\n", cpu_time_used);
    	fprintf(fptr3,"%d\n",epoch );
    }
    fclose(fptr);
    fclose(fptr2);
    fclose(fptr3);
    fclose(fptr4);
}

//dizide aslýnda arama yaptýk.Öncekilerle ayný olan kelime varsa sözlüðümüze eklemedik
int oncekilerleKiyas(int *indices, int anlikIndeks, int kiyasSayi) {
    int i = 0;
    for (i = 0; i < anlikIndeks; i++) {
        if (indices[i] == kiyasSayi) {
            return 1;
        }
    }
    return 0;
}
