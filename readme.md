# Simulace pohybů planet ve 2D

## Textový popis

Tento projekt se zabývá simulací pohybu planet (těles) ve 2D prostoru
(všechny objekty budou mít pro jednoduchost stejnou $z$ souřadnici).
Uvažujeme gravitační interakce mezi každou dvojicí těles v podobě
Newtonova gravitačního zákona $$F_{g}=G\frac{m_{1}m_{2}}{r^{2}},$$ kde
$G=6.674\cdot10^{-11}\left[m^{3}kg^{-1}s^{-2}\right]$ je gravitační
konstanta, $m_{1},m_{2}$ jsou váhy těles a $r$ je vzdálenost mezi nimi.
$F_{g}$ je velikost síly působící mezi dvojicí těles, směr síly je vždy
k druhému tělesu.

Cílem je implementovat numerickou aproximaci řešení pohybových rovnic
planet a vytvoření vizualizace tohoto pohybu. Pro simulaci bude použita
časová diskretizace, přičemž v každém časovém kroku je uvažováno
konstantní zrychlení $a=\frac{F_{g}}{m}$ způsobené gravitační silou
mezi každou dvojicí planet. $a$ je opět pouze velikost zrychlení, směr
je stejný jako směr síly.

Výstupem projektu bude vizualizace pohybu pomocí knihovny Matplotlib,
která zobrazuje pohyb planet v čase a zobrazuje jejich trajektorie.

## Funkcionality
-   načítání počátečních podmínek (polohy,
    rychlosti a hmotnosti) planet z json souboru
-   Vytvořit funkci pro výpočet zrychlení způsobené gravitační silou
    mezi všemi dvojicemi planet
-   Vykreslit aktuální polohy těles v čase
-   Ukládat polohy všech těles v čase pro pozdější vykreslení
    trajektorie
-   Pomocí série obrázků vytvořit animaci pohybu planet, která zobrazuje
    trajektorie a polohy planet v čase
-   Implementovat funkci pro uložení výsledné animace do video souboru
-   Vytvořit funkci pro generování náhodných scénářů simulace, která
    bude náhodně generovat počáteční podmínky planet (polohy, rychlosti
    a hmotnosti)
-   vyzkušet různé délky časového kroku (u planet např. hodina, den, týden, ...) a pozorovat kdy dojde k degradaci simulace 
