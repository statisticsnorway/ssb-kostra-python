# SSB Kostra Python

[![PyPI](https://img.shields.io/pypi/v/ssb-kostra-python.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/ssb-kostra-python.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/ssb-kostra-python)][pypi status]
[![License](https://img.shields.io/pypi/l/ssb-kostra-python)][license]

[![Documentation](https://github.com/statisticsnorway/ssb-kostra-python/actions/workflows/docs.yml/badge.svg)][documentation]
[![Tests](https://github.com/statisticsnorway/ssb-kostra-python/actions/workflows/tests.yml/badge.svg)][tests]
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=statisticsnorway_ssb-kostra-python&metric=coverage)][sonarcov]
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=statisticsnorway_ssb-kostra-python&metric=alert_status)][sonarquality]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)][poetry]

[pypi status]: https://pypi.org/project/ssb-kostra-python/
[documentation]: https://statisticsnorway.github.io/ssb-kostra-python
[tests]: https://github.com/statisticsnorway/ssb-kostra-python/actions?workflow=Tests
[sonarcov]: https://sonarcloud.io/summary/overall?id=statisticsnorway_ssb-kostra-python
[sonarquality]: https://sonarcloud.io/summary/overall?id=statisticsnorway_ssb-kostra-python
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black
[poetry]: https://python-poetry.org/

## Features

### Oppsummering
- Denne pakken er til bruk for KOSTRA-brukere og gjenskaper noen av funksjonene dere var vant med fra KOMPIS. Disse er ikke like smidige som dem som var tilgjengelige i KOMPIS, men de utfører i stor grad de samme oppgavene, og med litt tilvenning vil dere forhåpentligvis slippe å lage helt egne funksjoner for de vanligste operasjonene. I avsnittet under forklares hva funksjonene gjør.

### Funksjonene
| Oppgave | /ssb-kostra-python/src/ssb_kostra_python/mappe.funksjon |
|---|---|
| Konvertere variabeltype/runde av med valgt antall desimaler | ```python avrunding.konverter_dtypes``` |
| Hente/lage folkemengdefil 31.12.20XX for bydeler (samme oppsett som i KOMPIS). Du kan hente årganger gamle årganger, samt generere datasett for det kommende året i testperioden. Du må angi år, regionsnivå og hvorvidt dette skal gjelde reelle data eller testdata. | ```python hente_data_folkemengde.hente_data_folkemengde``` |
| Hente/lage folkemengdefil 31.12.20XX for kommuner (samme oppsett som i KOMPIS). Du kan hente årganger gamle årganger, samt generere datasett for det kommende året i testperioden. Du må angi år, regionsnivå og hvorvidt dette skal gjelde reelle data eller testdata. | ```python hente_data_folkemengde.hente_data_folkemengde``` |
| Hente/lage folkemengdefil 31.12.20XX for fylkeskommuner (samme oppsett som i KOMPIS). Du kan hente årganger gamle årganger, samt generere datasett for det kommende året i testperioden. Du må angi år, regionsnivå og hvorvidt dette skal gjelde reelle data eller testdata. | ```python hente_data_folkemengde.hente_data_folkemengde``` |
| Aggregere bydelsdata til aggregert KOSTRA-regioner (EAB) | ```python regionshierarki.hierarki``` (identifiserer riktig regionsnivå automatisk så lenge regionsvariabelen heter bydelsregion, kommuneregion eller fylkesregion, krever at du indentifiserer de øvrige klassifikasjonsvariablene i datasettet.) |
| Aggregere kommunedata til aggregerte KOSTRA-regioner (EAK, EAKUO, EKA, EKG) | ```python regionshierarki.hierarki``` (identifiserer riktig regionsnivå automatisk så lenge regionsvariabelen heter bydelsregion, kommuneregion eller fylkesregion, krever at du indentifiserer de øvrige klassifikasjonsvariablene i datasettet.) |
| Aggregere fylkeskommunedata til aggregerte KOSTRA-regioner (EAFK, EAFKUO) | ```python regionshierarki.hierarki``` (identifiserer riktig regionsnivå automatisk så lenge regionsvariabelen heter bydelsregion, kommuneregion eller fylkesregion, krever at du indentifiserer de øvrige klassifikasjonsvariablene i datasettet.) |
| Summere et datasett med kjønnsvariabel over kjønnene | ```python summere_kjonn.summere_over_kjonn``` |
| Summere et datasett som er fordelt på ettårige aldersgrupperinger til aggregerte KOSTRA-aldersgrupperinger | ```python summere_til_aldersgrupperinger.summere_til_aldersgrupperinger``` |
| Feste en kolonne med titler til KLASS-koder (f.eks 0301 - Oslo) | ```python titler_til_klasskoder.kodelister_navn``` |

## Eksempelark
Du finner eksempelark på [ssb-kostra-python/examples/](https://github.com/statisticsnorway/ssb-kostra-python/tree/main/examples). Der kan du kjøre gjennom funksjonene for å se hvordan det kan gjøres.

## Programvarekrav
For å bruke `ssb-kostra-python` må prosjektet ditt bruke:

- Python 3.12 eller nyere
- Poetry for håndtering av pakker og avhengigheter

Du kan kontrollere hvilken Python-versjon du bruker ved å skrive følgende
i terminalen:

```bash
python --version
```

## Tilganger til fellesbøtter
Funksjonene henter filer fra fellesbøtter hos **S312 Befolkningsstatistikk** og **S212 Offentlige finanser**.

Hos **S312** trenger du tilgang til:

**folketall** ---> **ssb-bef-statistikk-data-delt-folketall-prod**

Hos **S212** trenger du tilgang til:

**kostra-befolkning-delt** ---> **ssb-off-fin-data-delt-kostra-befolkning-delt-prod**

## Installasjon
Pakken installeres med Poetry. Kjør følgende kommando i terminalen fra
prosjektet der du ønsker å bruke pakken:

```bash
poetry add ssb-kostra-python
```

Du kan kontrollere om pakken er installert, og hvilken versjon du har,
med:

```bash
poetry show ssb-kostra-python
```

## Importere pakken
Merk at navnet som brukes når pakken installeres, og navnet som brukes
når pakken importeres i Python, er litt forskjellige.

Pakken installeres med:

```bash
poetry add ssb-kostra-python
```

Men i Python brukes understrek (`_`) i stedet for bindestrek (`-`):

```python
import ssb_kostra_python
```

Dette vil altså **ikke** fungere:

```python
import ssb-kostra-python
```

### Importere en funksjon

Funksjonene kan importeres direkte fra modulen der de ligger.

For eksempel kan funksjonen `hente_data_folkemengde` importeres slik:

```python
from ssb_kostra_python.hente_data_folkemengde import hente_data_folkemengde
```

Funksjonen kan deretter brukes på vanlig måte:

```python
folkemengde = hente_data_folkemengde(
    2024,
    "bydel",
    False,
)
```

Se tabellen over for en oversikt over hvilke moduler de forskjellige
funksjonene ligger i.

## Problemer med installasjon eller import

Dersom du får en feilmelding når du forsøker å importere eller bruke
`ssb-kostra-python`, kan du gå gjennom kontrollene nedenfor.

### Kort feilsøkingssjekk

Hvis pakken ikke fungerer som forventet, kan du starte med disse tre
kommandoene i terminalen:

```bash
python --version
poetry show ssb-kostra-python
poetry env info
```

Kjør deretter dette i notebooken:

```python
import ssb_kostra_python
import importlib.metadata

print("Pakke:", ssb_kostra_python.__file__)
print("Versjon:", importlib.metadata.version("ssb-kostra-python"))
```

Da får du kontrollert:

1. hvilken Python-versjon du bruker
2. hvilken versjon av `ssb-kostra-python` som er installert
3. hvilket Poetry-miljø som brukes
4. hvor Python faktisk finner `ssb_kostra_python`

Disse opplysningene vil ofte gjøre det mulig å finne årsaken dersom
installasjon eller import ikke fungerer.

### 1. Kontroller at pakken er installert

Skriv i terminalen:

```bash
poetry show ssb-kostra-python
```

Hvis pakken er installert, får du blant annet opp hvilken versjon du har.

Hvis pakken ikke er installert, installer den med:

```bash
poetry add ssb-kostra-python
```

### 2. Kontroller hvilken versjon som er installert

Du kan kontrollere versjonen i terminalen:

```bash
poetry show ssb-kostra-python
```

Du kan også kontrollere dette fra Python:

```python
import importlib.metadata

print(importlib.metadata.version("ssb-kostra-python"))
```

Dette kan være nyttig dersom en funksjon som finnes i en nyere versjon av
pakken, ikke finnes i miljøet ditt.

### 3. Oppdater en eldre versjon

Hvis du allerede har pakken installert, kan du forsøke å oppdatere den
med:

```bash
poetry update ssb-kostra-python
```

Kontroller deretter versjonen på nytt:

```bash
poetry show ssb-kostra-python
```

Hvis Poetry svarer:

```text
No dependencies to install or update
```

selv om det finnes en nyere versjon av `ssb-kostra-python`, kan
versjonskravet i prosjektets `pyproject.toml` hindre Poetry i å
installere den nye versjonen.

Kontroller hvilken versjon av ssb-kostra-python som er angitt i pyproject.toml. Du kan sammenligne denne
med den nyeste tilgjengelige versjonen på PyPI. Øverst på PyPI-siden vil du se versjonsnummeret ved siden
av pakketittelen ssb-kostra-python, for eksempel 0.1.0.

Hvis du da for eksempel ønsker å gå over til versjon 0.1.0, kan du eksplisitt
be Poetry om dette:

```bash
poetry add ssb-kostra-python@^0.1.0
```

### 4. Kontroller Python-versjonen

`ssb-kostra-python` krever Python 3.12 eller nyere.

Kontroller versjonen med:

```bash
python --version
```

Hvis du bruker en eldre Python-versjon, må Python-miljøet oppdateres før
den nyeste versjonen av `ssb-kostra-python` kan installeres.

### 5. Poetry sier at Python-versjonene ikke er kompatible

Det kan hende at du bruker Python 3.12 eller nyere, men likevel får en
feilmelding fra Poetry om Python-versjonen.

Da bør du kontrollere Python-kravet i prosjektets `pyproject.toml`.

Hvis det for eksempel står:

```toml
python = ">=3.11,<4.0"
```

betyr dette at prosjektet også skal kunne kjøres med Python 3.11.
`ssb-kostra-python` krever imidlertid Python 3.12 eller nyere.

Dersom prosjektet ditt ikke trenger å støtte Python 3.11, må Python-kravet
i `pyproject.toml` oppdateres til å kreve minst Python 3.12.

Det kan for eksempel se slik ut:

```toml
python = ">=3.12,<4.0"
```

Er du usikker på om Python-kravet for prosjektet kan endres, bør du
avklare dette før du gjør endringen.

Forsøk deretter å installere eller oppdatere pakken på nytt.

### 6. Kontroller hvilket Python-miljø som brukes

Det er mulig å ha flere Python-miljøer samtidig. Da kan pakken være
installert i ett miljø, mens notebooken din bruker et annet.

Du kan se hvor `ssb_kostra_python` hentes fra ved å kjøre:

```python
import ssb_kostra_python

print(ssb_kostra_python.__file__)
```

Resultatet vil for eksempel kunne se slik ut:

```text
/.../.venv/lib/python3.13/site-packages/ssb_kostra_python/__init__.py
```

Du kan også undersøke Poetry-miljøet fra terminalen:

```bash
poetry env info
```

Hvis miljøene ikke stemmer overens, må du sørge for at notebooken bruker
Python-miljøet der pakken er installert.

### 7. Restart kernel etter installasjon eller oppdatering

Hvis du bruker JupyterLab og nettopp har installert eller oppdatert
pakken, kan det være nødvendig å restarte kernelen før den nye versjonen
blir tilgjengelig.

Velg:

**Kernel → Restart Kernel**

og kjør deretter importen på nytt.


## Bruk
På [ssb-kostra-python/examples/](https://github.com/statisticsnorway/ssb-kostra-python/tree/main/examples) finner
du eksempel-notebooks som gjennomgår bruken av de ulike funksjonene.
Please see the [Reference Guide] for details.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_SSB Kostra Python_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [Statistics Norway]'s [SSB PyPI Template].

[statistics norway]: https://www.ssb.no/en
[pypi]: https://pypi.org/
[ssb pypi template]: https://github.com/statisticsnorway/ssb-pypitemplate
[file an issue]: https://github.com/statisticsnorway/ssb-kostra-python/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/statisticsnorway/ssb-kostra-python/blob/main/LICENSE
[contributor guide]: https://github.com/statisticsnorway/ssb-kostra-python/blob/main/CONTRIBUTING.md
[reference guide]: https://statisticsnorway.github.io/ssb-kostra-python/reference.html
