"""Tests for the ParameterStore class."""

import pytest

import pybamm
from pybamm.parameters.parameter_store import (
    ParameterCategory,
    ParameterStore,
    _detect_category,
    _parse_units,
)


class TestParameterStore:
    """Test the ParameterStore class."""

    def test_init_empty(self):
        store = ParameterStore()
        assert len(store) == 0

    def test_init_with_data(self):
        store = ParameterStore({"a": 1, "b": 2})
        assert store["a"] == 1
        assert store["b"] == 2
        assert len(store) == 2

    def test_getitem_setitem(self):
        store = ParameterStore()
        store["key"] = 42
        assert store["key"] == 42

    def test_getitem_missing_key(self):
        store = ParameterStore({"a": 1})
        with pytest.raises(KeyError, match="Parameter 'b' not found"):
            store["b"]

    def test_delitem(self):
        store = ParameterStore({"a": 1, "b": 2})
        del store["a"]
        assert "a" not in store
        assert "b" in store

    def test_contains(self):
        store = ParameterStore({"a": 1})
        assert "a" in store
        assert "b" not in store

    def test_iter(self):
        store = ParameterStore({"a": 1, "b": 2})
        keys = list(store)
        assert "a" in keys
        assert "b" in keys

    def test_len(self):
        store = ParameterStore({"a": 1, "b": 2, "c": 3})
        assert len(store) == 3

    def test_get_with_default(self):
        store = ParameterStore({"a": 1})
        assert store.get("a") == 1
        assert store.get("b", 99) == 99
        assert store.get("c") is None

    def test_set_allow_new_true(self):
        store = ParameterStore({"a": 1})
        store.set("b", 2, allow_new=True)
        assert store["b"] == 2

    def test_set_allow_new_false(self):
        store = ParameterStore({"a": 1})
        with pytest.raises(KeyError, match="does not exist"):
            store.set("b", 2, allow_new=False)

    def test_set_allow_new_false_existing(self):
        store = ParameterStore({"a": 1})
        store.set("a", 10, allow_new=False)
        assert store["a"] == 10

    def test_update_allow_new_true(self):
        store = ParameterStore({"a": 1})
        store.update({"a": 10, "b": 2}, allow_new=True)
        assert store["a"] == 10
        assert store["b"] == 2

    def test_update_allow_new_false(self):
        store = ParameterStore({"a": 1})
        with pytest.raises(KeyError, match="does not exist"):
            store.update({"a": 10, "b": 2}, allow_new=False)

    def test_update_conflict_raise(self):
        store = ParameterStore({"a": 1})
        with pytest.raises(ValueError, match="already exists"):
            store.update({"a": 2}, conflict="raise")

    def test_update_conflict_warn(self):
        store = ParameterStore({"a": 1})
        with pytest.warns(match="already exists"):
            store.update({"a": 2}, conflict="warn")
        assert store["a"] == 2

    def test_update_conflict_ignore(self):
        store = ParameterStore({"a": 1})
        store.update({"a": 2}, conflict="ignore")
        assert store["a"] == 2

    def test_keys_values_items(self):
        store = ParameterStore({"a": 1, "b": 2})
        assert list(store.keys()) == ["a", "b"]
        assert list(store.values()) == [1, 2]
        assert list(store.items()) == [("a", 1), ("b", 2)]

    def test_pop(self):
        store = ParameterStore({"a": 1, "b": 2})
        val = store.pop("a")
        assert val == 1
        assert "a" not in store

    def test_copy(self):
        store = ParameterStore({"a": 1})
        store_copy = store.copy()
        store_copy["a"] = 99
        assert store["a"] == 1  # Original unchanged

    def test_to_dict(self):
        store = ParameterStore({"a": 1, "b": 2})
        d = store.to_dict()
        assert d == {"a": 1, "b": 2}
        assert isinstance(d, dict)


class TestParameterInfo:
    """Test the ParameterInfo dataclass and get_info method."""

    def test_get_info_scalar(self):
        store = ParameterStore({"Temperature [K]": 298.15})
        info = store.get_info("Temperature [K]")
        assert info.name == "Temperature [K]"
        assert info.value == 298.15
        assert info.units == "K"
        assert info.is_function is False
        assert info.is_input is False

    def test_get_info_function(self):
        def my_func(x):
            return x * 2

        store = ParameterStore({"My function [A]": my_func})
        info = store.get_info("My function [A]")
        assert info.is_function is True
        assert info.units == "A"

    def test_get_info_input_parameter(self):
        store = ParameterStore({"My input [V]": pybamm.InputParameter("My input [V]")})
        info = store.get_info("My input [V]")
        assert info.is_input is True

    def test_get_info_no_units(self):
        store = ParameterStore({"Some parameter": 42})
        info = store.get_info("Some parameter")
        assert info.units is None


class TestParseUnits:
    """Test the _parse_units helper function."""

    def test_parse_units_basic(self):
        assert _parse_units("Temperature [K]") == "K"
        assert _parse_units("Concentration [mol.m-3]") == "mol.m-3"
        assert _parse_units("Current function [A]") == "A"

    def test_parse_units_no_units(self):
        assert _parse_units("Some parameter") is None
        assert _parse_units("Another one") is None


class TestDetectCategory:
    """Test the _detect_category helper function."""

    def test_detect_negative_electrode(self):
        assert (
            _detect_category("Maximum concentration in negative electrode [mol.m-3]")
            == "negative electrode"
        )

    def test_detect_positive_electrode(self):
        assert (
            _detect_category("Positive electrode thickness [m]") == "positive electrode"
        )

    def test_detect_electrolyte(self):
        assert _detect_category("Electrolyte conductivity [S.m-1]") == "electrolyte"

    def test_detect_thermal(self):
        assert _detect_category("Reference temperature [K]") == "thermal"

    def test_detect_other(self):
        assert _detect_category("Some random parameter") == "other"


class TestParameterCategory:
    """Test the ParameterCategory enum."""

    def test_category_values(self):
        assert ParameterCategory.NEGATIVE_ELECTRODE.value == "negative electrode"
        assert ParameterCategory.POSITIVE_ELECTRODE.value == "positive electrode"
        assert ParameterCategory.THERMAL.value == "thermal"


class TestListByCategory:
    """Test the list_by_category method."""

    def test_list_by_category_string(self):
        store = ParameterStore(
            {
                "Negative electrode thickness [m]": 0.001,
                "Positive electrode thickness [m]": 0.001,
                "Temperature [K]": 298.15,
            }
        )
        neg_params = store.list_by_category("negative electrode")
        assert "Negative electrode thickness [m]" in neg_params
        assert "Positive electrode thickness [m]" not in neg_params

    def test_list_by_category_enum(self):
        store = ParameterStore(
            {
                "Negative electrode thickness [m]": 0.001,
                "Temperature [K]": 298.15,
            }
        )
        neg_params = store.list_by_category(ParameterCategory.NEGATIVE_ELECTRODE)
        assert "Negative electrode thickness [m]" in neg_params


class TestCategories:
    """Test the categories method."""

    def test_categories(self):
        store = ParameterStore(
            {
                "Negative electrode thickness [m]": 0.001,
                "Positive electrode thickness [m]": 0.001,
                "Temperature [K]": 298.15,
            }
        )
        cats = store.categories()
        assert "negative electrode" in cats
        assert "positive electrode" in cats
        assert "thermal" in cats


class TestParameterDiff:
    """Test the diff method and ParameterDiff dataclass."""

    def test_diff_added(self):
        store1 = ParameterStore({"a": 1})
        store2 = ParameterStore({"a": 1, "b": 2})
        diff = store1.diff(store2)
        assert diff.added == {"b": 2}
        assert diff.removed == {}
        assert diff.changed == {}

    def test_diff_removed(self):
        store1 = ParameterStore({"a": 1, "b": 2})
        store2 = ParameterStore({"a": 1})
        diff = store1.diff(store2)
        assert diff.added == {}
        assert diff.removed == {"b": 2}
        assert diff.changed == {}

    def test_diff_changed(self):
        store1 = ParameterStore({"a": 1, "b": 2})
        store2 = ParameterStore({"a": 1, "b": 3})
        diff = store1.diff(store2)
        assert diff.added == {}
        assert diff.removed == {}
        assert diff.changed == {"b": (2, 3)}

    def test_diff_mixed(self):
        store1 = ParameterStore({"a": 1, "b": 2})
        store2 = ParameterStore({"b": 3, "c": 4})
        diff = store1.diff(store2)
        assert diff.added == {"c": 4}
        assert diff.removed == {"a": 1}
        assert diff.changed == {"b": (2, 3)}

    def test_diff_empty(self):
        store1 = ParameterStore({"a": 1})
        store2 = ParameterStore({"a": 1})
        diff = store1.diff(store2)
        assert diff.added == {}
        assert diff.removed == {}
        assert diff.changed == {}

    def test_diff_with_rtol(self):
        """Test that small differences are ignored with relative tolerance."""
        store1 = ParameterStore({"x": 1.0, "y": 100.0})
        store2 = ParameterStore({"x": 1.0 + 1e-10, "y": 100.0 + 1e-8})

        # Without tolerance, both show as changed
        diff_exact = store1.diff(store2)
        assert "x" in diff_exact.changed
        assert "y" in diff_exact.changed

        # With tolerance, small differences are ignored
        diff_tol = store1.diff(store2, rtol=1e-6)
        assert "x" not in diff_tol.changed
        assert "y" not in diff_tol.changed

    def test_diff_rtol_respects_magnitude(self):
        """Test that rtol is relative to the magnitude of the values."""
        store1 = ParameterStore({"small": 1e-6, "large": 1e6})
        # Add 0.1% difference to each
        store2 = ParameterStore({"small": 1e-6 * 1.001, "large": 1e6 * 1.001})

        # With 0.01% tolerance, both should show as changed
        diff = store1.diff(store2, rtol=1e-4)
        assert "small" in diff.changed
        assert "large" in diff.changed

        # With 1% tolerance, both should be ignored
        diff = store1.diff(store2, rtol=0.01)
        assert "small" not in diff.changed
        assert "large" not in diff.changed

    def test_diff_numpy_arrays(self):
        """Test exact and approximate comparison of NumPy arrays."""
        import numpy as np

        store1 = ParameterStore({"values": np.array([1.0, 2.0])})
        store2 = ParameterStore({"values": np.array([1.0, 2.000001])})

        # Without tolerance, arrays should be reported as different.
        diff_exact = store1.diff(store2)
        assert "values" in diff_exact.changed

        # A small difference should be ignored with sufficient tolerance.
        diff_tolerant = store1.diff(store2, rtol=1e-5)
        assert diff_tolerant.changed == {}

        # A stricter tolerance should detect the difference.
        diff_strict = store1.diff(store2, rtol=1e-8)
        assert "values" in diff_strict.changed

    def test_diff_callable_identity(self):
        """Functions are compared by identity, not their output."""

        def first(x):
            return x * 2

        def second(x):
            return x * 2

        store1 = ParameterStore({
            "unchanged": first,
            "changed": first,
        })
        store2 = ParameterStore({
            "unchanged": first,
            "changed": second,
        })

        diff = store1.diff(store2)

        assert diff.added == {}
        assert diff.removed == {}
        assert diff.changed == {"changed": (first, second)}

    def test_diff_handles_equality_error(self):
        """Comparison errors should mark values as changed, not crash."""

        class Uncomparable:
            def __eq__(self, other):
                raise ValueError("Cannot compare these objects")

        first = Uncomparable()
        second = Uncomparable()

        store1 = ParameterStore({"problematic": first})
        store2 = ParameterStore({"problematic": second})

        diff = store1.diff(store2)

        assert set(diff.changed) == {"problematic"}
        assert diff.changed["problematic"][0] is first
        assert diff.changed["problematic"][1] is second

    def test_diff_incompatible_array_shapes(self):
        """Arrays with incompatible shapes should be reported as changed."""
        import numpy as np

        first = np.array([1.0, 2.0])
        second = np.array([1.0, 2.0, 3.0])

        store1 = ParameterStore({"values": first})
        store2 = ParameterStore({"values": second})

        # np.allclose cannot broadcast these shapes together.
        # ParameterStore should handle the error without crashing.
        diff = store1.diff(store2, rtol=1e-6)

        assert set(diff.changed) == {"values"}
        assert diff.changed["values"][0] is first
        assert diff.changed["values"][1] is second