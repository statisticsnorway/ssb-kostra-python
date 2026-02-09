import pandas as pd
import pytest

from ssb_kostra_python.regionshierarki import hierarki
from ssb_kostra_python.regionshierarki import mapping_bydeler_oslo
from ssb_kostra_python.regionshierarki import mapping_fra_fylkeskommune_til_kostraregion
from ssb_kostra_python.regionshierarki import mapping_fra_kommune_til_fylkeskommune
from ssb_kostra_python.regionshierarki import mapping_fra_kommune_til_landet

# =============================================================================
# SECTION 1: mapping_bydeler_oslo (BYDELER)
# =============================================================================


class FakeCodes_bb_eab:
    """Test double for the object returned by KlassClassification.get_codes(...)."""

    def __init__(self, pivot_df: pd.DataFrame):
        """Initialize fake with dataframe returned by pivot_level()."""
        self._pivot_df = pivot_df

    def pivot_level(self):
        return self._pivot_df


class FakeKlassClassification_bb_eab:
    """Test double for KlassClassification used by mapping_bydeler_oslo."""

    pivot_df = None  # set per test

    def __init__(self, klass_id, language="nb", include_future=True):
        """Initialize fake KlassClassification with expected constructor arguments."""
        self.klass_id = klass_id
        self.language = language
        self.include_future = include_future

    def get_codes(self, *args, **kwargs):
        """Return fake codes object for testing."""
        return FakeCodes_bb_eab(self.__class__.pivot_df)


class TestMappingBydelerOslo:
    """Tests for mapping_bydeler_oslo(year)."""

    def test_mapping_bydeler_oslo_filters_and_sets_to_EAB(self, mocker):
        """Checks if the function maps and filters correctly."""
        mocker.patch(
            "ssb_kostra_python.regionshierarki.KlassClassification",
            FakeKlassClassification_bb_eab,
        )
        FakeKlassClassification_bb_eab.pivot_df = pd.DataFrame(
            {
                "code_1": ["030101", "030116", "030117", "030102", "EAB", "030103"],
                "name_1": [
                    "A",
                    "B",
                    "C",
                    "D",
                    "Samlebydel",
                    "E",
                ],
            }
        )

        out = mapping_bydeler_oslo(year="2024")

        assert list(out.columns) == ["from", "to"]
        assert "030116" not in out["from"].tolist()
        assert "030117" not in out["from"].tolist()
        assert "EAB" not in out["from"].tolist()
        assert sorted(out["from"].tolist()) == sorted(["030101", "030102", "030103"])
        assert (out["to"] == "EAB").all()

    def test_mapping_bydeler_oslo_returns_empty_if_only_filtered_codes(self, mocker):
        """Verify behavior when all available codes are filtered out."""
        mocker.patch(
            "ssb_kostra_python.regionshierarki.KlassClassification",
            FakeKlassClassification_bb_eab,
        )
        FakeKlassClassification_bb_eab.pivot_df = pd.DataFrame(
            {
                "code_1": ["030116", "030117", "EAB"],
            }
        )

        out = mapping_bydeler_oslo(year=2024)

        assert list(out.columns) == ["from", "to"]
        assert len(out) == 0


# =============================================================================
# SECTION 2: mapping_fra_kommune_til_landet (KOMMUNER -> REGIONSGRUPPERINGER)
# =============================================================================


class FakeKlassCorrespondence_kk_eak:
    """Fake KlassCorrespondence that provides a `.data` DataFrame."""

    def __init__(
        self, source_classification_id, target_classification_id, from_date, to_date
    ):
        """Initializes an instance of the class with specified parameters."""
        self.source_classification_id = source_classification_id
        self.target_classification_id = target_classification_id
        self.from_date = from_date
        self.to_date = to_date

        if str(target_classification_id) == "104":  # kommune -> fylke
            self.data = pd.DataFrame(
                {
                    "sourceCode": ["0301", "5001", "9999"],
                    "targetCode": ["0300", "5000", "9999"],
                }
            )
        elif str(target_classification_id) == "112":  # kommune -> KOSTRA-gruppe
            self.data = pd.DataFrame(
                {
                    "sourceCode": ["0301", "5001", "9999"],
                    "targetCode": ["K1", "K2", "KX"],
                }
            )
        else:
            self.data = pd.DataFrame(columns=["sourceCode", "targetCode"])


class FakeCodes_kk_eak:
    """Fake codes object providing pivot_level(), used by FakeKlassClassification_kk_eak."""

    def __init__(self, pivot_df: pd.DataFrame):
        """Initializes an instance of the class with specified parameters."""
        self._pivot_df = pivot_df

    def pivot_level(self):
        return self._pivot_df


class FakeKlassClassification_kk_eak:
    """Fake KlassClassification returning a FakeCodes object."""

    pivot_df = None

    def __init__(self, klass_id, language="nb", include_future=True):
        """Initializes an instance of the class with specified parameters."""
        self.klass_id = klass_id
        self.language = language
        self.include_future = include_future

    def get_codes(self, *args, **kwargs):
        return FakeCodes_kk_eak(self.__class__.pivot_df)


class TestMappingFraKommuneTilLandet:
    """Tests for mapping_fra_kommune_til_landet(year)."""

    def test_mapping_fra_kommune_til_landet_happy_path(self, mocker):
        """Validate the main invariants of mapping_fra_kommune_til_landet."""
        mocker.patch(
            "ssb_kostra_python.regionshierarki.KlassCorrespondence",
            FakeKlassCorrespondence_kk_eak,
        )
        mocker.patch(
            "ssb_kostra_python.regionshierarki.KlassClassification",
            FakeKlassClassification_kk_eak,
        )
        FakeKlassClassification_kk_eak.pivot_df = pd.DataFrame(
            {
                "code_1": ["0301", "5001", "9999"],
                "name_1": ["Oslo", "Trondheim", "Ugyldig"],
            }
        )

        out = mapping_fra_kommune_til_landet(year="2024")

        assert list(out.columns) == ["from", "to"]
        assert not (out["from"] == "9999").any()
        assert out["from"].str.len().eq(4).all()

        assert ((out["from"] == "0301") & (out["to"] == "EKA03")).any()
        assert ((out["from"] == "5001") & (out["to"] == "EKA50")).any()

        assert ((out["from"] == "0301") & (out["to"] == "K1")).any()
        assert ((out["from"] == "5001") & (out["to"] == "K2")).any()

        assert ((out["from"] == "0301") & (out["to"] == "EAK")).any()
        assert ((out["from"] == "5001") & (out["to"] == "EAK")).any()

        assert not ((out["from"] == "0301") & (out["to"] == "EAKUO")).any()
        assert ((out["from"] == "5001") & (out["to"] == "EAKUO")).any()

    def test_from_is_zero_padded_when_short(self, mocker):
        """Specifically validate the zero-padding behavior."""
        mocker.patch(
            "ssb_kostra_python.regionshierarki.KlassCorrespondence",
            FakeKlassCorrespondence_kk_eak,
        )
        mocker.patch(
            "ssb_kostra_python.regionshierarki.KlassClassification",
            FakeKlassClassification_kk_eak,
        )
        FakeKlassClassification_kk_eak.pivot_df = pd.DataFrame(
            {
                "code_1": ["301", "9999"],
                "name_1": ["ShortCode", "Ugyldig"],
            }
        )

        out = mapping_fra_kommune_til_landet(year=2024)

        assert (out["from"] == "0301").any()
        assert not (out["from"] == "301").any()


# =============================================================================
# SECTION 3: mapping_fra_kommune_til_fylkeskommune (KOMMUNER -> FYLKESKOMMUNER)
# =============================================================================


class FakeKlassCorrespondence_kk_fk:
    """Minimal fake for KlassCorrespondence(...).data used in mapping_fra_kommune_til_fylkeskommune."""

    def __init__(self, *args, **kwargs):
        """Initializes an instance of the class with specified parameters."""
        self.data = pd.DataFrame(
            {
                "sourceCode": ["0301", "9999", "1103", "301"],
                "targetCode": ["0300", "9999", "1100", "11"],
            }
        )


class TestMappingFraKommuneTilFylkeskommune:
    """Tests for mapping_fra_kommune_til_fylkeskommune(year)."""

    def test_mapping_filters_renames_and_pads(self, mocker):
        """Verify filtering and padding."""
        mocker.patch(
            "ssb_kostra_python.regionshierarki.KlassCorrespondence",
            FakeKlassCorrespondence_kk_fk,
        )
        out = mapping_fra_kommune_til_fylkeskommune("2024")

        assert list(out.columns) == ["from", "to"]
        assert not (out["from"] == "9999").any()
        assert out["from"].map(lambda x: isinstance(x, str) and len(x) == 4).all()
        assert out["to"].map(lambda x: isinstance(x, str) and len(x) == 4).all()

        assert ((out["from"] == "0301") & (out["to"] == "0300")).any()
        assert ((out["from"] == "0301") & (out["to"] == "0011")).any()


# =============================================================================
# SECTION 4: mapping_fra_fylkeskommune_til_kostraregion (FYLKE -> KOSTRA-GRUPPER)
# =============================================================================


class FakeKlassCorrespondence_fk_eafk:
    """Fake KlassCorrespondence that returns deterministic mapping rows for fylkes codes."""

    def __init__(
        self, source_classification_id, target_classification_id, from_date, to_date
    ):
        """Initializes an instance of the class with specified parameters."""
        self.data = pd.DataFrame(
            {
                "sourceCode": ["0300", "4200", "9900"],
                "targetCode": ["R1", "R2", "R9"],
            }
        )


class FakeCodes_fk_eafk:
    def __init__(self, pivot_df: pd.DataFrame):
        """Initializes an instance of the class with specified parameters."""
        self._pivot_df = pivot_df

    def pivot_level(self):
        return self._pivot_df


class FakeKlassClassification_fk_eafk:
    """Fake KlassClassification providing fylkes codes via pivot_level()."""

    pivot_df = None

    def __init__(self, klass_id, language="nb", include_future=True):
        """Initializes an instance of the class with specified parameters."""
        self.klass_id = klass_id

    def get_codes(self, *args, **kwargs):
        return FakeCodes_fk_eafk(self.__class__.pivot_df)


class TestMappingFraFylkeskommuneTilKostraregion:
    """Tests for mapping_fra_fylkeskommune_til_kostraregion(year)."""

    def test_mapping_fra_fylkeskommune_til_kostraregion_happy_path(self, mocker):
        """Verify fylke to kostraregion mapping."""
        mocker.patch(
            "ssb_kostra_python.regionshierarki.KlassCorrespondence",
            FakeKlassCorrespondence_fk_eafk,
        )
        mocker.patch(
            "ssb_kostra_python.regionshierarki.KlassClassification",
            FakeKlassClassification_fk_eafk,
        )
        FakeKlassClassification_fk_eafk.pivot_df = pd.DataFrame(
            {
                "code_1": ["0300", "4200", "9900"],
                "name_1": ["Oslo", "Agder", "Ugyldig"],
            }
        )

        out = mapping_fra_fylkeskommune_til_kostraregion(year="2024")

        assert list(out.columns) == ["from", "to"]
        assert ((out["from"] == "0300") & (out["to"] == "R1")).any()
        assert ((out["from"] == "4200") & (out["to"] == "R2")).any()
        assert ((out["from"] == "9900") & (out["to"] == "R9")).any()

        assert ((out["from"] == "0300") & (out["to"] == "EAFK")).any()
        assert ((out["from"] == "4200") & (out["to"] == "EAFK")).any()
        assert not ((out["from"] == "9900") & (out["to"] == "EAFK")).any()

        assert not ((out["from"] == "0300") & (out["to"] == "EAFKUO")).any()
        assert ((out["from"] == "4200") & (out["to"] == "EAFKUO")).any()
        assert not ((out["from"] == "9900") & (out["to"] == "EAFKUO")).any()

        assert len(out) == 6


# =============================================================================
# SECTION 5: hierarki(df, aggregeringstype=...)
# =============================================================================


class TestHierarki:
    """Tests for the main `hierarki` function."""

    def test_raises_if_more_than_one_periode(self):
        """Hierarki expects exactly one unique periode."""
        df = pd.DataFrame(
            {
                "periode": ["2024", "2025"],
                "kommuneregion": ["0301", "0301"],
                "personer": [1, 2],
            }
        )
        with pytest.raises(KeyError, match="Mer enn 1 periode"):
            hierarki(df)

    def test_raises_if_no_region_column(self):
        """Hierarki expects exactly one valid region column to exist."""
        df = pd.DataFrame({"periode": ["2025"], "personer": [1]})
        with pytest.raises(ValueError, match="Fant ingen gyldig regionkolonne"):
            hierarki(df)

    def test_raises_if_multiple_region_columns(self):
        """Hierarki expects exactly one region column."""
        df = pd.DataFrame(
            {
                "periode": ["2025"],
                "kommuneregion": ["0301"],
                "fylkesregion": ["0300"],
                "personer": [1],
            }
        )
        with pytest.raises(ValueError, match="Fant flere regionskolonner"):
            hierarki(df)

    def test_raises_if_inconsistent_aggregeringstype(self):
        """Verify aggregeringstype consistency."""
        df = pd.DataFrame(
            {
                "periode": ["2025"],
                "fylkesregion": ["0300"],
                "personer": [1],
            }
        )
        with pytest.raises(ValueError, match="Inkonsekvent valg"):
            hierarki(df, aggregeringstype="kommune_til_landet")

    def test_kommune_til_landet_default_appends_aggregated_rows(self, mocker):
        """Verify default kommune to landet aggregation."""
        df = pd.DataFrame(
            {
                "periode": ["2025", "2025"],
                "kommuneregion": ["0301", "0301"],
                "alder": ["001", "002"],
                "personer": [10, 20],
            }
        )

        mock_map = mocker.patch(
            "ssb_kostra_python.regionshierarki.mapping_fra_kommune_til_landet"
        )
        mock_map.return_value = pd.DataFrame({"from": ["0301"], "to": ["EAK"]})
        mock_definer = mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.definere_klassifikasjonsvariable"
        )
        mock_definer.return_value = (
            ["periode", "kommuneregion", "alder"],
            ["personer"],
        )

        out = hierarki(df)

        assert len(out) == 4
        assert (
            (out["kommuneregion"] == "EAK")
            & (out["alder"] == "001")
            & (out["personer"] == 10)
        ).any()
        assert (
            (out["kommuneregion"] == "EAK")
            & (out["alder"] == "002")
            & (out["personer"] == 20)
        ).any()

        mock_map.assert_called_once()
        mock_definer.assert_called_once()

    def test_kommune_til_fylkeskommune_filters_to_endswith_00_and_renames(self, mocker):
        """Verify aggregation and column renaming."""
        df = pd.DataFrame(
            {
                "periode": ["2025", "2025"],
                "kommuneregion": ["0301", "5001"],
                "personer": [10, 20],
            }
        )

        mock_map = mocker.patch(
            "ssb_kostra_python.regionshierarki.mapping_fra_kommune_til_fylkeskommune"
        )
        mock_map.return_value = pd.DataFrame(
            {
                "from": ["0301", "5001"],
                "to": ["0300", "5000"],
            }
        )
        mock_definer = mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.definere_klassifikasjonsvariable"
        )
        mock_definer.return_value = (["periode", "kommuneregion"], ["personer"])

        out = hierarki(df, aggregeringstype="kommune_til_fylkeskommune")

        assert "fylkesregion" in out.columns
        assert "kommuneregion" not in out.columns
        assert out["fylkesregion"].str.endswith("00").all()

        mock_map.assert_called_once()
        mock_definer.assert_called_once()

    def test_fylkeskommune_til_kostraregion(self, mocker):
        """Verify fylkesregion to kostraregion aggregation."""
        df = pd.DataFrame(
            {
                "periode": ["2025", "2025"],
                "fylkesregion": ["0300", "4200"],
                "personer": [5, 7],
            }
        )

        mock_map = mocker.patch(
            "ssb_kostra_python.regionshierarki.mapping_fra_fylkeskommune_til_kostraregion"
        )
        mock_map.return_value = pd.DataFrame(
            {"from": ["0300", "4200"], "to": ["KFK1", "KFK2"]}
        )
        mock_definer = mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.definere_klassifikasjonsvariable"
        )
        mock_definer.return_value = (["periode", "fylkesregion"], ["personer"])

        out = hierarki(df)

        assert ((out["fylkesregion"] == "KFK1") & (out["personer"] == 5)).any()
        assert ((out["fylkesregion"] == "KFK2") & (out["personer"] == 7)).any()

        mock_map.assert_called_once()
        mock_definer.assert_called_once()

    def test_bydeler_til_EAB(self, mocker):
        """Verify bydel aggregation into EAB."""
        df = pd.DataFrame(
            {
                "periode": ["2025", "2025"],
                "bydelsregion": ["030101", "030102"],
                "personer": [3, 4],
            }
        )

        mock_map = mocker.patch(
            "ssb_kostra_python.regionshierarki.mapping_bydeler_oslo"
        )
        mock_map.return_value = pd.DataFrame(
            {"from": ["030101", "030102"], "to": ["EAB", "EAB"]}
        )
        mock_definer = mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.definere_klassifikasjonsvariable"
        )
        mock_definer.return_value = (["periode", "bydelsregion"], ["personer"])

        out = hierarki(df)

        eab_rows = out[out["bydelsregion"] == "EAB"]
        assert len(eab_rows) == 1
        assert eab_rows["personer"].iloc[0] == 7

        mock_map.assert_called_once()
        mock_definer.assert_called_once()
