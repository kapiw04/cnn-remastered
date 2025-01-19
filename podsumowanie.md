# Podsumowanie spotkania

Materiały z których korzystałem to [dokumentacja pytorch](https://pytorch.org) oraz [moje skromne repozytorium](https://github.com/kapiw04/cnn-mnist). 

## Wstępne info

Zanim zaczniemy pracę z danymi od koła (które dostaniemy znacznie później) musimy sobie potrenować korzystanie z bibliotek pythona i ogólnie tworzenie sieci neuronowych. Ustaliliśmy, że skorzystamy z biblioteki `pytorch`, która jest nieźle odokumentowana i raczej przystępna dla początkujących. Jako wstępne zadanie w ramach koła, każdy z nas **wytrenuje sieć neuronową klasyfikującą pisane cyfry**. Posłuży nam do tego zbiór danych MNIST. Jest on dostępny bardzo szeroko publicznie. Jest na tyle common, że można go zdobyć już z poziomu pytorcha:
```py
from torchvision.datasets.mnist import MNIST

train = MNIST(root="data", train=True, download=True)  # 60000 obrazków
test = MNIST(root="data", train=False, download=True)  # 10000 obrazków
```

Zbenefitujemy z tego, ponieważ:

- ogarniemy jak działają ze sobą komponenty biblioteki
- zobaczymy jakie są typowe probelmy z analizą danych i jak je rozwiązywac
- dowiemy się jak trenować konwolucyjną sieć neuronową
- fajny projekcik do portfolio i CV

## Trochę o obrazkach i wymiarach

O obrazie będziemy teraz myśleli jak o macierzach. Każda komórka reprezentuje 1 piksel. Żeby opisać obrazek czarnobiały wystarczy, że w każdym pikselu będzie 1 wartość od 0 (czarny) do 255 (biały). Wartości pomiędzy to odcienie szarości.

Gdy chcemy kolorowy obraz, musimy przyjąć aż 3 wymiary. Każdy kolor jest reprezentowany przez 3 wartości: czerwony, zielony i niebieski (RGB).

Przykładowo:

$$
\begin{pmatrix}
  (255, 0, 0) & (0, 255, 0) \\
  (0, 0, 255) & (255, 255, 255) \\
\end{pmatrix}
$$

to obrazek 2x2:

$$
\begin{pmatrix}
  czerwony & zielony \\
  niebieski & biały \\
\end{pmatrix}
$$

Obrazki mogą być też 6-, 16- albo 1000- wymiarowe. Dla naszych oczu nic to nie zmieni, ale dla komputera pozwoli odkryć cenne informacje i feature'y, które pozwolą mu rozpoznawać cyferki



## Dane treningowe

W pytorchu do manipulacji zbiorami danych używa się klasy `torch.utils.data.Dataset`. W kodzie wyżej `train` i `test` są właśnie obiektami tej klasy :)
Potem bardzo wygodnie się z nich korzysta w procesie trenowania. Dzielimy dane na trenignowe i testowe, aby upewnić się, że nasz model nie "oszukuje" gdy sprawdzamy jego accuracy - chcemy wiedzieć jak dobrze działa na nowych danych (testowych), na podstawie danych które widział (treningowe).

### Opcjonalnie 
`torch.utils.data.Dataloader` to dodatkowe "opakowanie" do `Dataset`u. Największą zaletą jest to, że można wczytywać dane podczas treningu za pomocą batchy. Jednym z benefitów jest przyspieszenie procesu treningowego oraz ugeneralizowanie danych, co przyczyni się do performence'u modelu. 

Więcej [tutaj](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#preparing-your-data-for-training-with-dataloaders)

## Klasa modelu w torchu

Model w pytorchu wrzuca się do klasy. [Przykład z dokumentacji](https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html#torch-nn-module-and-torch-nn-parameter); [Przykład z mojego repo](https://github.com/kapiw04/cnn-mnist/blob/main/models.py) (ten `class CNN(nn.Module)`)

Te klasy mają 2 podstawowe funkcje: `__init__` (konstruktor) oraz `forward`. 

W `__init__` definiujemy warstwy w naszym modelu.
Np:
```py
self.conv1 = nn.Conv2d(1, conv_1_size, kernel_size)
```
Oznacza warstwę konwolucyjną, która przyjmuje (kolejno argumenty): 
 - **1** - input. 1 obrazek. 
 - **conv_1_size** - wymiar wartswy. u mnie chyba 6. czyli outputem jest obraz 6wymiarowy. 
 - **kernel_size** - rozmiar kernela możesz przyjąć 3 lub 5

Nie musisz dogłębnie rozumieć co to za operacja. Generalnie jest to pewna funkcja: Przyjmuje obraz i zwraca obraz. Model będzie się starał tak dobrać wartości w kernelu, aby starać się "rozumieć" obraz który mu podamy.

[Tutaj](https://cdn.arstechnica.net/wp-content/uploads/2018/10/Screen-Shot-2018-10-12-at-4.45.46-PM.png) pokazane jest jak wyglądają te kernele dla innej sieci (AlexNet). Widać, że model szuka krawędzi w różnych kierunkach (te czarno-białe kernele), albo przejścia między kolorami. 

Zauważ, że rozmiary warstw muszą się "zazębiać". Tzn. conv1 **zwraca** ci obraz 6-wymiarowy, więc covn2 musi **przyjmować** obraz 6-wymiarowy.

`forward` to funckja mówiąca co ma się dziać przy przejściu przez architekturę sieci (warstwa po wartsiwe)

```py
    # Z linku do dokumentacji
    def forward(self, x): # x to input - w naszym przypadku to byłby obraz, tutaj idk może jakiś wektor np. (1, 2, 3, 4)
        x = self.linear1(x) # x przepuszczony przez wcześniej (w __init__) zdefiniowaną warstwę liniową
        x = self.activation(x) # funkcja aktywacji - chcemy pozbyć się liniowości z modelu
        x = self.linear2(x) # to samo co 2 linie wyżej
        x = self.softmax(x) # z warstwy linear2 chcemy dostać prawdopodobieństwa na daną klasę (patrz później)
        return x
```

Trochę więcej o funckji aktywacji i po co jest: [tutaj](https://www.reddit.com/r/MLQuestions/comments/13j1g1y/comment/jkctk3x/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) :)

Softmax to wartswa, która ma "przetłumaczyć" to co ma do powiedzenia model na faktyczny interesujący nas output. Np możemy otrzymać z funckcji softmax wektor
(0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0). Oznacza to że **prawdopodobieństwo** według modelu na to że podany input jest cyfrą 2 wynosi 100%. 

## Trenowanie modelu

W tym miejscu myślę, że najlepiej sprawdzi się rzut oka do [dokumentacji](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html). Jest tez tam filmik w którym gość tłumaczy o co chodzi. 

[Tutaj](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-training-loop) sam fragment pętli trenowania.


Ważnymi pojęciami są:
- **funkcja straty (loss function)** - funkcja mówiąca nam jak bardzo model się pomylił.
- **optimizier** - funkcja która szuka minimum funkcji straty

Podczas trwania pętli szukamy takich parametrów (np. wartości w kernelu), które dają najmniejszą wartość funkcji straty - czyli najlepsze rezultaty. 

Nie ma najlepszej [funckji straty](https://pytorch.org/docs/stable/optim.html#algorithms) ani [optimizera](https://pytorch.org/docs/stable/optim.html#algorithms). W torchu jest ich cała masa. Polecam poeksperymentować z kilkoma i zobaczyć, która sprawdza się najlepiej (można np. zacząć od tych z najbardziej cool nazwą B) ).

## Dobieranie parametrów

Tak samo jak wyżej. Nie ma najlepszych. Na jakieś trzeba się zdecydować, poeksperymentować i iść dalej. 

Co można dostosować:
 - funkcja straty
 - optimizer
 - architektura sieci (dodać/odjąć wartswy, zmienić ich rozmiary etc.)
 - rozmiar batchy
 - learning rate (The learning rate is a hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated [source](https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/))
 - momentum (Momentum in neural networks is a parameter optimization technique that accelerates gradient descent by adding a fraction of the previous update to the current update. [source](https://www.geeksforgeeks.org/what-is-momentum-in-neural-network/))
 - w jakim stosunku podzielić dane treningowe i testowe (domyślnie 1:6, można próbować 10%, 20%...)
 - ilość epok treningowych (epoka to przejście po całym zestawie treningowych danych)
 - inne rzeczy o których nie pamiętam
 - alterowanie obrazka (resize, crop, normalizacja, [tu implementacja](https://pytorch.org/vision/0.9/transforms.html#scriptable-transforms))
 - basically: wszystko co masz w kodzie można jakoś zmienić

# Podsumowanie

Mam nadzieję, że nie jest to za duży infodump. Jak coś chętnie odpowiem na pytania, albo usiądę np. na discorda w wolnym czasie i pomoge z implementacją. 
Najważniejsze jest to, że torch baardzo ułatwia ten proces. Jest już dużo gotowych algorytmów do skorzystania i można z nich korzystać nawet jak się nie rozumie w 100% jak działają. 

Ustaliliśmy deadline na task do końca ferii. Potem porównamy swoje rozwiązania i zobaczymy, czyj model najlepiej rozpozna cyfry napisane przez tajemniczego gościa.

Good Luck!