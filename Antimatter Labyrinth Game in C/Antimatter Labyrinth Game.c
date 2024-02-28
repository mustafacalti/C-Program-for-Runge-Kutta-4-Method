#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

typedef struct {
	char isim[50];
	char soyisim[50];
	char kullaniciAdi[50];
	char sifre[50];
}kullanici;					// Kullan�c�m�z�n bilgilerini tuttu�umuz struct.

typedef struct {
	char kullaniciAdi[50];
	float puan;
}kullaniciSkor;              //Skor dosyas�ndan okunan veriyi isim ve puan olarak ay�rmka i�in olu�turdu�um struct.


//Kullandi[im fonks'yonlarin prototipleri.
kullanici kullaniciOlusturma(const char *dosyaAdi);
int oncekilerleKiyasKullaniciAdi(kullanici* kullaniciListesi,char *isim,char *soyisim,int kullaniciSayisi);
kullanici girisYapma(const char *dosyaAdi);
char **oyunAlaniOlusturma();
void oyunAlaniBastirma(char **oyunAlanim,int n, int m);
void txtdenOkuma(const char *dosyaAdi, char ***oyunAlani,int *n,int *m);
void kullaniciKoordinatBulma(char **oyunAlani,int *x,int *y,int);
void bosluklardanKurtarma(char **oyunAlani, int n,int *m);
float hamleYapma(char **oyunAlani, int n, int m, int *x, int *y,int *flag,char *,int *,int,int,time_t *);
float sagaHamle(char **oyunAlani,int *x,int *y,char *parcaciklar,int *parcacikSayisi,int *flag);
float solaHamle(char **oyunAlani,int *x,int *y,char *parcaciklar,int *parcacikSayisi,int *flag);
float yukariHamle(char **oyunAlani,int *x,int *y,char *parcaciklar,int *parcacikSayisi,int *flag);
float asagiHamle(char **oyunAlani,int *x,int *y,char *parcaciklar,int *parcacikSayisi,int *flag);
void parcacikDizisiGuncelleme(char *parcaciklar,int *parcacikSayisi,char parcacik);
void karadeligeUlasildi(int *flag);
float oyunSonu(char *parcaciklar,int *parcacikSayisi,int *flag);
int skorlariKarsilastir(char *dosya,float skor, kullanici kullanici);
void oyunuBaslat(time_t *baslangicZamani);
void skorlariBastir(char *dosya);
void bilgileriGoster();

int main(){
	int s=0,n,m,x,y;
	float skor=0; // Baslangicta skor 0.
	char **oyunAlani;
	kullanici kullanici;
	int k;
	int secim;
	int secim2;
	int secim3; // Switch case , do while ya da ifler icin kullandigim degiskenler.
	int yenidenKullanma;
		
	printf("Oyuna hosgeldiniz! \n");
	printf("Yeni kullanici olusturmak icin 1'i, var olan bir kullaniciya giris yapmak icin ise 2'yi tuslayiniz: ");
	scanf("%d",&secim);
	switch (secim){
		
		case 1:
			kullanici = kullaniciOlusturma("dosyamiz.bin"); // dosyamiz.bin binary sekilde kullanici bilgilerini tuttugumuz dosya, case 1 yeni kullanici, case 2 ise giris yapma ekrani.
			break;
			
		case 2:
			kullanici = girisYapma("dosyamiz.bin");
			break;
	}
	do{
	system("cls");
	

	
	system("cls");
	printf("Oyunumuzda en yuksek 5 skoru yapmis kisileri ve bu skorlari gormek icin 1'i,\nOyunun nasil oynanacagi hakkindaki bilgileri goruntulemek icin 2'yi\nOyuna baslamak icin 3'u tuslayiniz:  "); // Ana men�..
	scanf("%d",&secim2);
	switch(secim2){
		case 1:
			skorlariBastir("skordosyasi.txt");
			break;
		case 2:
			bilgileriGoster();
			break;
		case 3:
			printf("\nHazir haritalardan biriyle oynamak icin 1'i, \nistediginiz bir haritayi yuklemek icin ise 2'yi tuslayiniz(Lutfen haritanizi oyunAlani isimli bir txt dosyasina kaydedin): ");  // Kullan�c� haz�r haritalardan birisini ya da oyunAlani.txt i�erisine kaydetmi� oldu�u kendi oyun alan�yla oynaabilir.
			scanf("%d",&secim);
			if(secim == 1){
				printf("1. harita : \n");
				printf("C 0 0 e 0\n0 P E p 0\nP e P X e\nE 0 e K P\n1 P 0 P e\n1 0 K P 0\n0 0 e P 0");
				printf("\n\n2. harita : \n");
				printf("C p e 0 0 0 0\n0 1 E 0 0 P 0\nP e P X 0 0 0\nE 0 e 0 P 0 e\n1 P 0 P e 1 0\nK 0 1 P 0 P 0\n0 0 e P 0 e 0");    // Haz�r haritalar�n bast�r�lmas�.
				printf("\n\n3. harita : \n");
				printf("0 e 0 0 E 0 C 0\n0 0 0 e P p 0 0\np E P X 0 0 0 0\nE 0 e e 0 e 0 0\n1 P 0 P e K 0 0\n1 P 0 P 0 P 0 e\n0 e 0 P 0 e e 0");
				printf("\n\n4. harita : \n");
				printf("C 0 0 e 0\n0 P E p 0\nP e P X e\nE 0 e 0 P\n1 P 0 P e\n1 0 K P 0\n0 0 e P 0");
				printf("\n Seciminizi yapiniz: ");
				scanf("%d",&secim3);
				if(secim3 == 1){
					FILE *oyun = fopen("oyunAlani.txt","w");
					fprintf(oyun, "C 0 0 e 0\n0 P E p 0\nP e P X e\nE 0 e K P\n1 P 0 P e\n1 0 K P 0\n0 0 e P 0");
					fclose(oyun);
				}
				else if(secim3 == 2){
					FILE *oyun = fopen("oyunAlani.txt","w");
					fprintf(oyun, "C p e 0 0 0 0\n0 1 E 0 0 P 0\nP e P X 0 0 0\nE 0 e 0 P 0 e\n1 P 0 P e 1 0\nK 0 1 P 0 P 0\n0 0 e P 0 e 0");    // Haz�r haritalar� dosyaya yaz�p kullan�c�ya oynatmak i�in fprintf fonksiyonu yard�m�yla oyun alan� olu�turulmas�.
					fclose(oyun);
				}
				else if(secim3 == 3){
					FILE *oyun = fopen("oyunAlani.txt","w");
					fprintf(oyun, "0 e 0 0 E 0 C 0\n0 0 0 e P p 0 0\np E P X 0 0 0 0\nE 0 e e 0 e 0 0\n1 P 0 P e K 0 0\n1 P 0 P 0 P 0 e\n0 e 0 P 0 e e 0");
					fclose(oyun);
				}
				else if(secim3 == 4){
					FILE *oyun = fopen("oyunAlani.txt","w");
					fprintf(oyun, "C 0 0 e 0\n0 P E p 0\nP e P X e\nE 0 e 0 P\n1 P 0 P e\n1 0 1 K 0\n0 0 e P 0");
					fclose(oyun);
				}
				txtdenOkuma("oyunAlani.txt",&oyunAlani,&n,&m); // Kullan�c�n�n sec�m�nden sonra txt dosyas�ndan oyunAlani matrisine okuma yap�p oyun alan�n�n olu�turulmas�.
				if(oyunAlani[0][1] == ' '){
					bosluklardanKurtarma(oyunAlani,n,&m); // E�er kullan�c� haritay� girerken '0EpE' yerine '0 E p E' �eklinde girmi�se matrise yaz�lm�� verinin indislerini d�zeltmek ve kodu kolayla�t�rmak i�in bir kontorl. 
				}
				kullaniciKoordinatBulma(oyunAlani,&x,&y,m);   // Oyun alani hazirlandiktan sonra kullanicinin koordinatlarini bulup oyunu onun uzerinden oynatacagiz. Burda da koordinatlari buluyoruz.
			}
			else if(secim == 2){
				txtdenOkuma("oyunAlani.txt",&oyunAlani,&n,&m); // Yukaridaki islemlerin aynisi.. harita sectirmeden bir onceki (varolan) haritay� y�kler ya da yeni olu�turulup txt dosyas�n�n i�ine at�lm�� olan haritay� oynat�r.
				if(oyunAlani[0][1] == ' '){
					bosluklardanKurtarma(oyunAlani,n,&m);
				}
				kullaniciKoordinatBulma(oyunAlani,&x,&y,m);
			}
			char *parcaciklar;
			parcaciklar = (char *)malloc(1 * sizeof(char)); // oyundayken elde etti�imiz p E P ve e par�ac�klar�n� tutup sonradan saymak i�in olu�turdu�um dizi.
			int parcacikSayisi=0; // Her oyunun ba��nda toplanan par�ac�k say�s�n�n s�f�rlanmas�.
			int flag=3; // Oyunu sonland�rmak i�in kulland���m flag.
			int xcpy = x; // Ba�lang�� koordinatlar�n� ba�ka fonksiyonlarda kullanmak �zere kopyalad�m.
			int ycpy = y;
			time_t baslangicZamani = time(NULL); // Kullan�c�n�n belirli bir zamanda oyunu sonland�r�p puan almas� ya da s�renin d���na ��k�p oyundan at�lmas�n� sa�lamak i�in ba�latt���m zaman.
			while(flag == 3){
				skor = hamleYapma(oyunAlani, n, m, &x, &y,&flag,parcaciklar,&parcacikSayisi,xcpy,ycpy,&baslangicZamani); // Oyunu oynatt���m fonksiyon.
			}
			if(skor>1100) // E�er oyun sonunda kullan�c� ba�ar�l� �ekilde kar��t par�a �retemediyse burada yapt��� skor 0'lan�yor.
			skor = 0;
			skorlariKarsilastir("skordosyasi.txt",skor,kullanici);  // Her oyun sonunda elimdeki kullan�c�n�n skoru ile bundan �nceki oyunlarda tuttu�um 5 en y�ksek skoru kar��la�t�r�p burdaki en y�ksek 5 skoru tutan fonksiyon.
			break;
		}
		printf("\n Ana menuye donmek icin lutfen 1'i, cikmak icin baska bir tusu tuslayiniz : "); // Kullan�c�n�n ayn� kullan�c� olarak oyunu tekrar tekrar oynayabilmesini sa�layan kod.(Yeni kullan�c� giri�i i�in ��k�p girmek laz�m).
		scanf("%d",&yenidenKullanma);
	}while(yenidenKullanma == 1);
	
	return 0;
	
}

void swap(kullaniciSkor *a, kullaniciSkor *b) { // 6 kullaniciyi kiyaslayip selection sort ediyorum. Bunun icin call-by refference kullanan bir swap fonksiyonu.
    kullaniciSkor temp = *a;
    *a = *b;
    *b = temp;
}

void selectionSort(kullaniciSkor *kullanicilar) {
	int i,j;
    for (i = 0; i < 5; i++) { 
        int maxIndex = i;
        for (j = i + 1; j < 6; j++) {
            if (kullanicilar[j].puan > kullanicilar[maxIndex].puan) {   // Klasik bir selection sort uygulamas�. Her ad�mda b�y�kten k����e do�ru s�ralama yap�yor. max�ndex burada iterasyonlardaki en b�y�k de�erin indisini tutuyor. 
                maxIndex = j;
            }
        }
        if (maxIndex != i) {
            swap(&kullanicilar[i], &kullanicilar[maxIndex]);
        }
    }
}

int skorlariKarsilastir(char *dosya,float skor, kullanici kullanici){
	FILE *ilkSkorlar = fopen(dosya,"r"); // Skorlarin tutuldugu dosyayi aciyoruz.
	if(ilkSkorlar == NULL){
		fclose(ilkSkorlar);
		FILE *skorlar = fopen(dosya,"w"); 
		int i;
    	for(i=0; i<5; i++){
    		fprintf(skorlar, "a 0\n"); // Eger dosya null'sa selectionSort fonksiyonuna d�zg�n bir �ekilde dizimizi g�nderebilmek i�in dosyan�n i�ine 0 olan 5 adet skor ve a isminde kullan�c� at�yoruz.
    	}
    	fclose(skorlar);
    	FILE *ilkSkorlar = fopen(dosya,"r"); // Dosya tekrardan okuma modunda a��l�yor.
	}
	kullaniciSkor kullanicilar[6]; // dosyadan diziye okuyup selection sort'ta diziyi s�ralamak i�in 6 elemanl�k bir kullan�c�lar dizisi olu�turdum. Dizi kullaniciSkor struct�ndan 6 eleman tutuyor.
	int i=0; // Dizinin ilk eleman�ndan itibaren eleman eklemek i�in kulland���m�z de�i�ken.
	while (fscanf(ilkSkorlar, "%s %f", &kullanicilar[i].kullaniciAdi, &kullanicilar[i].puan) == 2) // Dosyadan kullan�c�lar dizisindeki her bir indisin kullaniciAdi ve puan degiskenlerine atama yap�l�yor.
	i++;
	kullaniciSkor kullanicii; // Dosyadan bundan �nceki 5 en y�ksek kullan�c�y� okuduk. �imdi ise parametre olarak ald���m�z kullan�c�y� ve onun skorunu atamak i�in olu�turdu�umuz ge�ici de�i�keni olu�turuyoruz.
	strcpy(kullanicii.kullaniciAdi,kullanici.kullaniciAdi);
	kullanicii.puan = skor; // Skoru ve kullaniciadini kullanicii isimli de�i�kenimize atad�k.
	kullanicilar[5] = kullanicii; // Kullan�c�m�z� dizinin i�ine atad�k.
	selectionSort(kullanicilar); // Selection sort'a dizimizi yollad�k. 
	fclose(ilkSkorlar); 
	
	FILE *skorlar = fopen(dosya,"w");
	for (i = 0; i < 5; i++) {
        fprintf(skorlar, "%s %f\n", kullanicilar[i].kullaniciAdi, kullanicilar[i].puan); // Siralanmis dizinin ilk 5 elemanini en yuksek skor basta olmak uzere dosyamiza yaziyoruz.
    }
    fclose(skorlar);
}
void bosluklardanKurtarma(char **oyunAlani, int n,int *m){
	int i,j;
	for(i=0; i<n; i++)
	for(j=2; j<(*m); j+=2){
		oyunAlani[i][j/2] = oyunAlani[i][j];   // E�er bo�luklu bir �ekilde oyun alan� verilmi�se 2'�erli bir �ekilde ilerleyerek her eleman� bulundu�u indisin yar�s�na kayd�rarak oyun alan�n� oynanabilir hale getiriyoruz.
	}
	(*m) = (*m) / 2; // S�tun say�s�n� yar�ya b�l�yoruz.
}

void kullaniciKoordinatBulma(char **oyunAlani,int *x,int *y,int m){
	int i=0,j=0;
	while(oyunAlani[i][j] != 'X'){ // Kullan�c�n�n indisini buldu�umuz fonksiyon.
		if(j==m){ // Sat�r sonuna gelindi�inde alt sat�ra ge�mek i�in kontrol.
			j=0;
			i++;
		}else
		j++;
	}
	*x = i;
	*y = j;
}

void txtdenOkuma(const char *dosyaAdi, char ***oyunAlani, int *n, int *m) {
    FILE *dosya = fopen(dosyaAdi, "r");
    if (dosya == NULL) {
        fprintf(stderr, "Dosya a�ilamadi.\n");
        exit(EXIT_FAILURE);
    }

    char buffer[100]; //Bir sat�r� okuyaca��z. Bu sat�r i�in gerekli olan buffer de�i�kenimiz.
    int i;

    *n = 0; // �lk sat�r ve s�tun say�s� de�erlerimizi 0'l�yoruz.
    *m = 0; //
    *oyunAlani = NULL;

    while (fgets(buffer, sizeof(buffer), dosya) != NULL) { // Bir sat�r� okuyoruz.
        (*n)++; // Sat�r say�m�z� bir artt�r�yoruz.
        *oyunAlani = (char **)realloc(*oyunAlani, (*n) * sizeof(char *)); // Oyun alan� matrisini bir sat�r geni�letiyoruz.
        if (*oyunAlani == NULL) {
            fprintf(stderr, "Bellek genisletme hatasi.\n");
            exit(EXIT_FAILURE);
        }

        (*m) = strlen(buffer); // s�tun say�s�n� buffer de�i�keninin uzunlu�una at�yoruz.
        (*oyunAlani)[*n - 1] = (char *)malloc((*m + 1) * sizeof(char));  // s�tun say�s� kadar yer a��yoruz sat�ra.

        if ((*oyunAlani)[*n - 1] == NULL) {
            fprintf(stderr, "Bellek tahsisi hatasi.\n");
            exit(EXIT_FAILURE);
        }

        strcpy((*oyunAlani)[*n - 1], buffer); // A�t���m�z sat�ra okudu�umuz sat�r� yaz�yoruz.
        
    }
    fclose(dosya);
}

int oncekilerleKiyas(const char *dosyaAdi, kullanici *kullanicim) {
    FILE *dosyaOkuma = fopen(dosyaAdi, "rb");
    if (dosyaOkuma == NULL) {
        fprintf(stderr, "Dosya acilamadi.\n");
        return -1; // Hata durumunu belirtmek i�in -1 d�nd�r�l�yor.
    }
	
    kullanici kullaniciTmp;
	fseek(dosyaOkuma, 0, SEEK_SET);
    while (fread(&kullaniciTmp, sizeof(kullanici), 1, dosyaOkuma) == 1) { // Kullan�c� bilgilerinin tutuldu�u .bin uzant�l� dosyay� okuyup burdaki veriyi kullan�c� struct� olarak okuyup kullan�c�tmp de�i�kenine at�yoruz.
    	
        if ( strcmp(kullaniciTmp.kullaniciAdi,kullanicim->kullaniciAdi) == 0)  { // Sat�rdan okudu�umuz de�i�kenimizin kullan�c� ad�yla kontrol etmek i�in parametre olarak yollad���m�z kullan�c� de�i�keninin kullan�c� isimlerini k�yasl�yoruz.
            fclose(dosyaOkuma);
            return 1; // Eslesme bulundu
        }
    }
    fclose(dosyaOkuma);
    return 0; // Eslesme bulunamadi
}

int sifreDenemesi(const char *dosyaAdi, kullanici *kullanicim){
	FILE *dosyaOkuma = fopen(dosyaAdi, "rb");
    if (dosyaOkuma == NULL) {
        fprintf(stderr, "Dosya acilamadi.\n");
        return -1; // Hata durumunu belirtmek i�in -1 d�nd�r�lebilir
    }
	
    kullanici kullaniciTmp;
    
	fseek(dosyaOkuma, 0, SEEK_SET);
    while (fread(&kullaniciTmp, sizeof(kullanici), 1, dosyaOkuma) == 1) {
    	
        if ( strcmp(kullaniciTmp.kullaniciAdi,kullanicim->kullaniciAdi) == 0 && strcmp(kullaniciTmp.sifre,kullanicim->sifre) == 0 )  { // Sifre verilerini yukaridaki kullaniciadi kiyasi gibi kiyaslayip duruma gore 0 ya da 1 donduruyoruz.
            fclose(dosyaOkuma);
            return 1; // Eslesme bulundu
        }
    }
    fclose(dosyaOkuma);
    return 0; // Eslesme bulunamadi
}

kullanici kullaniciOlusturma(const char *dosyaAdi){
	FILE *dosyaYazma = fopen(dosyaAdi, "ab");
	if(dosyaYazma == NULL){
		printf("Dosya acilamadi..");
		//return -1;
	}
		char gecici[50];
		kullanici kullanicim;
		printf("Kullanici ismi: ");
		scanf("%s",&gecici);
		strcpy(kullanicim.isim,gecici);
		printf("Kullanici soyismi: ");
		scanf("%s",&gecici);                                // Kullanicidan bilgilerini istiyoruz.
		strcpy(kullanicim.soyisim,gecici);
		int flag=1;
		do{
		printf("Kullanici adi: ");
		scanf("%s",&gecici);
		strcpy(kullanicim.kullaniciAdi,gecici);                  //Kullanici olustururken e�siz bir kullan�c� ad� girene kadar kullan�c�dan kullan�c� ad� istiyoruz.
		if (oncekilerleKiyas(dosyaAdi, &kullanicim)) {
        	printf("Ayni kullanici bulundu. Tekrar deneyin.\n");
			flag=0;
    	}
		else{
			flag=1;      
		}}while(flag==0);
		printf("Lutfen sifrenizi girin: ");
		scanf("%s",&gecici);
		strcpy(kullanicim.sifre,gecici);
		fwrite(&kullanicim, sizeof(kullanici), 1, dosyaYazma);                    //Kullan�c�n�n bilgilerini struct �ekilde dosyaya yaz�yoruz.
		printf("Yeni kullanici kaydi basariyla gerceklesti!\n");
		fclose(dosyaYazma);
		return kullanicim;	
}

kullanici girisYapma(const char *dosyaAdi){
	FILE *dosyaOkuma = fopen(dosyaAdi, "rb");   // Kullan�c� bilgilerini tuttu�umuz dosyay� binary okuma modunda a��yoruz.
	char gecici[50];
	char sifre[50];                    //Ge�ici de�i�kenlerimiz. 
	int flag;
	kullanici girisYapma;              //Kontrol edece�imiz ve e�er do�ru giri� sa�lan�rsa return edece�imiz kullan�c�.
	do{
		printf("Lutfen giris yapmak istediginiz kullanici adinizi girin: ");    
		scanf("%s",&gecici);
		strcpy(girisYapma.kullaniciAdi,gecici);
		if (oncekilerleKiyas(dosyaAdi, &girisYapma)) {                        // Kullan�c�dan ald���m�z kullan�c� ad�n� kontrol ediyoruz. 
        	//printf("Kullanici bulundu..\n");
			flag=1;
    	}
		else{
			printf("Bu kullanici adi ile kayit olunmamistir. Lutfen kullanici adinizi kontrol ediniz..\n");  //Kullan�c�ya d�nd�rd���m hata mesaj�.
			flag=0;
		}}while(flag==0);
	flag=1;
	do{
		printf("Lutfen sifrenizi girin: ");    
		scanf("%s",&sifre);
		strcpy(girisYapma.sifre,sifre);
		if (sifreDenemesi(dosyaAdi, &girisYapma)) { //Denenecek olan �ifreyi fonksiyonumuza g�nderiyoruz.
        	printf("Sifreniz dogru..\n");
			flag=0;
    	}
		else{
			printf("Yanlis sifre..\n");
			flag=1;
		}}while(flag==1);
		printf("Giris yapma islemi basariyla gerceklesti!");
		return girisYapma;   //Kullan�c�y� return ediyoruz.
		
}



void oyunAlaniBastirma(char **oyunAlanim, int n, int m){
    printf("---------Oyun Alaniniz---------\n\n");
    int i, j;
    for(i = 0; i < n; i++){
        for(j = 0; j < m; j++){
        	if(oyunAlanim[i][j] == 'X'){
        		printf("\033[1;33m%c\033[0m ", oyunAlanim[i][j]); 
			}
			else if(oyunAlanim[i][j] == 'K'){
				printf("\033[1;31m%c\033[0m ",oyunAlanim[i][j]);
			}
			else if(oyunAlanim[i][j] == 'C'){
				printf("\033[1;34m%c\033[0m ",oyunAlanim[i][j]);
			}
			else if (oyunAlanim[i][j] == 'P' || oyunAlanim[i][j] == 'p') {
			    printf("\033[1;35m%c\033[0m ", oyunAlanim[i][j]); 
			}
			else if (oyunAlanim[i][j] == 'E' || oyunAlanim[i][j] == 'e') {
    			printf("\033[1;36m%c\033[0m ", oyunAlanim[i][j]); 
			}
			else{
            printf("%c ", oyunAlanim[i][j]);
        }
    }
        printf("\n");
    }
}

void toplananParcacikGosterme(char *parcaciklar, int *parcacikSayisi) {
    int i;
    
    if (*parcacikSayisi != 0) { //Parcaciklar toplandiktan sonra kulanicinin gormesi icin bastirilacak 
        printf("Toplanan Parcaciklar:\n"); //En az 1 parcacikl toplandiktan sonra kulanicinin gormesi icin bastirilacak 
        

        for (i = 0; i < *parcacikSayisi; i++) {
            switch (parcaciklar[i]) {
                case 'p':
                    printf("P- ");
                    break;
                case 'P':
                    printf("P+ ");
                    break;
                case 'e':
                    printf("e- ");
                    break;
                case 'E':
                    printf("e+ ");
                    break;
                default:
                    break;
            }
        }
        printf("\n");
    }
}




float hamleYapma(char **oyunAlani, int n, int m, int *x, int *y, int *flag, char *parcaciklar,int *parcacikSayisi,int xcpy,int ycpy,time_t *baslangicZamani) {
    system("cls");   //Her hamlede ekran sifirlanir.
    float skor=0; //Skoru sifirliyorum.
	if(xcpy != *x || ycpy != *y) 
	oyunAlani[xcpy][ycpy] = 'G';  //Kullaniciya baslangic noktasini gostermek icin baslangic koordinatlarini G yaptim.
	
    oyunAlaniBastirma(oyunAlani, n, m); //Oyun alanini bastiriyorum.
	toplananParcacikGosterme(parcaciklar,parcacikSayisi);
	time_t simdikiZaman = time(NULL);   //�lk hmaleden bu yana ne kadar zaman ge�ti�ini hesaplamak i�in zaman� ba�lat�yorum.
    int gecenSure = difftime(simdikiZaman, *baslangicZamani);
    int kalanSure = 120 - gecenSure; 
    printf("\n\n Kalan sureniz : %d\n",kalanSure); //Kullan�c�ya ne kadar sures� kald�g�n� gosteriyorum.
    if(gecenSure > 120){
    	printf("Size ayrilan sure icerisinde oyunu tamamlayamadiniz! Ana ekrana donduruluyorsunuz.."); // S�re doldu.
    	*flag = 6;
	}
	printf("\nLutfen hamlenizi yapiniz..");
    
    
	char a = getch(); //Kullan�c�dan yapt��� tu� hamlesini al�yoruz.

    if (a == 77){
    	if(!(oyunAlani[*x][(*y)+1] == '1' || (*y) == m-1 )){ //S�n�r ve 1(Duvar) Kontrol�
        	skor = sagaHamle(oyunAlani, x, y,parcaciklar,parcacikSayisi,flag); 
			}
		}
    else if (a == 75){
    	if(!(oyunAlani[*x][(*y)-1] == '1' || (*y) == 0 )){
        	skor = solaHamle(oyunAlani, x, y,parcaciklar,parcacikSayisi,flag);
			}
		}
    else if (a == 72){
    	if(!(oyunAlani[(*x)-1][(*y)] == '1' || (*x) == 0 )){
        skor = yukariHamle(oyunAlani, x, y,parcaciklar,parcacikSayisi,flag); //Bas�lan tu�a g�re hamleler.
		}
	}
    else if (a == 80){
    	if(!(oyunAlani[(*x)+1][(*y)] == '1' || (*x) == n-1 )){
        skor = asagiHamle(oyunAlani, x, y,parcaciklar,parcacikSayisi,flag);
		}
	}
    else if (a == 27) { //Kullan�c� esc tu�una basarsa ��k��.
        printf("\n OYUNDAN CIKILIYOR... ");
        (*flag)=6;
    }
    
    if(skor != 0){
    	if(skor == -1){
    	(*flag)=6;} //Oyunu bitirme.
    	if(skor > 0){
    		(*flag)=6;
    			time_t gecenSuree = difftime(simdikiZaman, *baslangicZamani);
    			if(skor < 10){
    				skor = (skor * 80 / 100)  +  ((float)(120-gecenSuree) * 20 / 10000) * 10 * (m * n / 20); //Kullan�c�n�n yapt��� skor.
    				printf("\n\nSkor degerin: %f",skor);}
			return skor;
    		
	}
}
     sleep(0.1);
}

void parcacikDizisiGuncelleme(char *parcaciklar,int *parcacikSayisi,char parcacik){
	(*parcacikSayisi)++;
	char *gecici;
	parcaciklar = realloc(parcaciklar, (*parcacikSayisi) * sizeof(char)); //Dizinin boyutunu artt�r�p elde edilen par�ac��� sonuna ekliyoruz.
	parcaciklar[(*parcacikSayisi)-1] = parcacik; 
	
}


float sagaHamle(char **oyunAlani,int *x,int *y,char *parcaciklar,int *parcacikSayisi,int *flag){
	float skor;
	oyunAlani[*x][*y] = '0'; //Kullan�c�n�n hamleden �nceki bulundu�u yeri 0 yap�yoruz.
 	(*y)++;
 	if(oyunAlani[*x][*y] == 'K'){
 		karadeligeUlasildi(flag); //Karadeli�e ula��ld�.
 		return -1;
	 }
	 else if(oyunAlani[*x][*y] == 'C'){      
 		skor = oyunSonu(parcaciklar,parcacikSayisi,flag); //Oyunu bitiriyoruz ��k��a ula��ld�ysa.
 		return skor;
	 }
	 else if(oyunAlani[*x][*y] != '0' && oyunAlani[*x][*y] != 'G' && oyunAlani[*x][*y] != 'X'){    //E�er par�ac�k topland�ysa burada par�ac�k dizisini g�ncelliyoruz.
 		parcacikDizisiGuncelleme(parcaciklar,parcacikSayisi,oyunAlani[*x][*y]);
	 }
 	oyunAlani[*x][*y] = 'X';   //Kullan�c�n�n oldu�u yeri(hamleden sonraki) X yap�yoruz.  Ayn� �eyler di�er t�m hamleler i�in ge�erli.
 	return 0;
}
void karadeligeUlasildi(int *flag){
	system("cls");
	printf("Karadelige ulasildigi icin oyun sonlandirilmistir!");  //Karadeli�e ula��ld� uyar�s�.
}

float solaHamle(char **oyunAlani,int *x,int *y,char *parcaciklar,int *parcacikSayisi,int *flag){
	float skor;
	oyunAlani[*x][*y] = '0';
 	(*y)--;
 	if(oyunAlani[*x][*y] == 'K'){
 		karadeligeUlasildi(flag);
 		return -1;
	 }
	 else if(oyunAlani[*x][*y] == 'C'){
 		skor = oyunSonu(parcaciklar,parcacikSayisi,flag);    //Buradaki gerekli yorum sat�rlar� sa�a hamlede yaz�lm��t�r.
	 	return skor;
	 }
 	else if(oyunAlani[*x][*y] != '0' && oyunAlani[*x][*y] != 'G' && (oyunAlani[*x][*y] != 'X')){
 		parcacikDizisiGuncelleme(parcaciklar,parcacikSayisi,oyunAlani[*x][*y]);
	 }
 	oyunAlani[*x][*y] = 'X';
 	return 0;
}

float asagiHamle(char **oyunAlani,int *x,int *y,char *parcaciklar,int *parcacikSayisi,int *flag){
	float skor;
	oyunAlani[*x][*y] = '0';
 	(*x)++;
 	if(oyunAlani[*x][*y] == 'K'){
 		karadeligeUlasildi(flag);
 		return -1;
	 }
	 else if(oyunAlani[*x][*y] == 'C'){
 		skor = oyunSonu(parcaciklar,parcacikSayisi,flag);       //Buradaki gerekli yorum sat�rlar� sa�a hamlede yaz�lm��t�r.
 		return skor;
	 }
 	else if(oyunAlani[*x][*y] != '0' && oyunAlani[*x][*y] != 'G' && (oyunAlani[*x][*y] != 'X')){
 		parcacikDizisiGuncelleme(parcaciklar,parcacikSayisi,oyunAlani[*x][*y]);
	 }
 	oyunAlani[*x][*y] = 'X';
 	return 0;
}

float yukariHamle(char **oyunAlani,int *x,int *y,char *parcaciklar,int *parcacikSayisi,int *flag){
	float skor;
	oyunAlani[*x][*y] = '0';
 	(*x)--;
 	if(oyunAlani[*x][*y] == 'K'){
 		karadeligeUlasildi(flag);
 		return -1;
	 }
	 else if(oyunAlani[*x][*y] == 'C'){
 		skor = oyunSonu(parcaciklar,parcacikSayisi,flag);   //Buradaki gerekli yorum sat�rlar� sa�a hamlede yaz�lm��t�r.
		return skor;	 
	 }
 	else if(oyunAlani[*x][*y] != '0' && oyunAlani[*x][*y] != 'G' && (oyunAlani[*x][*y] != 'X')){
 		parcacikDizisiGuncelleme(parcaciklar,parcacikSayisi,oyunAlani[*x][*y]);
	 }
 	oyunAlani[*x][*y] = 'X';
	return 0;
}

float oyunSonu(char *parcaciklar,int *parcacikSayisi,int *flag){
	system("cls");
	printf("Oyununuz basarili bir sekilde sonlandi..");
	int i,psayisi=0,Psayisi=0,esayisi=0,Esayisi=0,skor=1600;
	for(i=0; i<(*parcacikSayisi); i++){
		if(parcaciklar[i] == 'p')
		psayisi++;
		else if(parcaciklar[i] == 'P')    //Kullan�c�n�n toplad��� par�ac�klar dizisinde gezilerek par�ac�k tespit edilmi� ve say�s� artt�r�lm��t�r.
		Psayisi++;
		else if(parcaciklar[i] == 'e')
		esayisi++;
		else if(parcaciklar[i] == 'E')
		Esayisi++;
	}
	printf("Topladiginiz toplam parcacik sayisi : %d\np- sayisi: %d\np+ sayisi: %d\ne-sayisi: %d\ne+ sayisi: %d\n",psayisi+Psayisi+esayisi+Esayisi,psayisi,Psayisi,esayisi,Esayisi);  //Kullan�c�ya toplad�klar� par�ac�klar hakk�nda bilgi veriliyor.
	psayisi = psayisi - Psayisi;
	Esayisi = Esayisi - esayisi;
	if (psayisi <= 0 || Esayisi <= 0)      //E�er e+ say�s� e- say�s�ndan az ya da e�itse veya p- say�s� p+ say�s�ndan az ya da e�itse kar��t madde �retilemiyor.
	printf("Karsit madde uretilemedi.");
	else if(psayisi > Esayisi){  //E�er p- sayisi daha fazlaysa E+ say�s� kadar kar��t madde �retilir.
	printf("%d adet karsit madde uretildi.",Esayisi);
	skor = Esayisi;
	}else{ 
		printf("%d adet karsit madde uretildi.",psayisi);  //E�er e+ sayisi daha fazlaysa p- say�s� kadar kar��t madde �retilir.
		skor = psayisi;
	}
	return skor;  //Kullan�c�ya e+ say�s� ya da p- say�s� kadar skor d�nd�r�l�r.
}

void oyunuBaslat(time_t *baslangicZamani) {
    *baslangicZamani = time(NULL);    //Oyunu ba�lat�rken zaman� �l�meye yarayan ba�lang��.
}

void skorlariBastir(char *dosya){
	FILE *skordosyasi = fopen(dosya,"r");  //Skorlar� tuttu�umuz dosyay� a��yoruz.
	if(skordosyasi == NULL)
	printf("\nHenuz 0'dan yuksek puan alan cikmadi :( \n");
	kullaniciSkor kullanicilar[5];   //Dosyadan verileri okumak i�in olu�turulan ge�ici dizi.
	int i=0;
	while (fscanf(skordosyasi, "%s %f", &kullanicilar[i].kullaniciAdi, &kullanicilar[i].puan) == 2){
		if(kullanicilar[i].puan > 0) //Skoru 0'dan b�y�k olan kullan�c�lar� ekrana yazd�r�yoruz.
		printf("%d. Kullanicinin Skoru : %f, Kullanici Adi: %s\n",i+1,kullanicilar[i].puan,kullanicilar[i].kullaniciAdi);
		i++;
	}
}
void bilgileriGoster(){  // Kullan�c�ya oyunla ilgili bilgileri ve kurallar� g�steriyoruz.
	system("cls");
	printf("1. Labirentteki yollarda asagidaki parcaciklar bulunmaktadir.\n P+ : proton\ne- : elektron\nP- : karsit proton \ne+ : karsit elektron\n");
	printf("2. Karsit hidrojen uretebilmek icin labirentin cikisinda elinizde sadece P- ve e+ parcaciklari bulunmalidir.\n3. Bir parcacikla o parcacigin zit isaretli karsit par�acigi bir araya gelirse birbirini yok eder.\n");
	printf("4. Karadelikler K ile gosterilmistir. Karadeliklerin bulundugu hucrelerden gecildigi takdirde oyun sonlanir.\n5. 120 saniye(2 dakika) icinde kullanici cikisa ulasamazsa oyun sonlanir.");
	printf("6. 0 ile gosterilenler yol, 1 ile gosterilenler ise duvardir. Duvarlardan gecemezsiniz ve eger sinirlarda hamle yapmaya calisirsaniz hamleniz basarisiz olur.\n");
}
