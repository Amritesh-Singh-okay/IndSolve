"""
IndSolve — Programmatic Streamlit AppTest Test Suite
Validates that all views render and buttons execute without exceptions using Streamlit AppTest.
"""

import unittest
from streamlit.testing.v1 import AppTest


class TestStreamlitAppViews(unittest.TestCase):
    def test_scenario_lab_render_and_apply(self):
        """Validates that Scenario Lab renders and Apply scenario form button executes without exceptions."""
        at = AppTest.from_file("app.py", default_timeout=15)
        at.run()
        self.assertFalse(at.exception, f"Exceptions on mount: {at.exception}")
        
        # Click "Apply scenario" form submit button
        if len(at.button) > 0:
            at.button[0].click().run()
            self.assertFalse(at.exception, f"Exceptions after Apply scenario: {at.exception}")

    def test_model_explorer_render_and_solve(self):
        """Validates that Model Explorer renders and Solve button executes without exceptions."""
        at = AppTest.from_file("app.py", default_timeout=15)
        at.run()
        # Switch navigation to Model Explorer (index 1)
        at.radio[0].set_value("Model Explorer").run()
        self.assertFalse(at.exception, f"Exceptions on Model Explorer mount: {at.exception}")
        
        # Find and click Solve Optimization Model button
        solve_buttons = [b for b in at.button if "Solve Optimization Model" in b.label]
        if solve_buttons:
            solve_buttons[0].click().run()
            self.assertFalse(at.exception, f"Exceptions after Solve click: {at.exception}")

    def test_validation_render_and_run(self):
        """Validates that Validation Lab renders and Run Verification button executes without exceptions."""
        at = AppTest.from_file("app.py", default_timeout=15)
        at.run()
        # Switch navigation to Validation (index 2)
        at.radio[0].set_value("Validation").run()
        self.assertFalse(at.exception, f"Exceptions on Validation mount: {at.exception}")
        
        # Find and click Run Verification button
        verify_buttons = [b for b in at.button if "Run Verification" in b.label]
        if verify_buttons:
            verify_buttons[0].click().run()
            self.assertFalse(at.exception, f"Exceptions after Run Verification click: {at.exception}")

    def test_import_model_render_and_solve(self):
        """Validates that Import Model (MPS) renders and Solve Parsed MPS Model button executes without exceptions."""
        at = AppTest.from_file("app.py", default_timeout=15)
        at.run()
        # Switch navigation to Import Model (index 3)
        at.radio[0].set_value("Import Model").run()
        self.assertFalse(at.exception, f"Exceptions on Import Model mount: {at.exception}")
        
        # Find and click Solve Parsed MPS Model button
        mps_buttons = [b for b in at.button if "Solve Parsed MPS" in b.label]
        if mps_buttons:
            mps_buttons[0].click().run()
            self.assertFalse(at.exception, f"Exceptions after Solve Parsed MPS click: {at.exception}")


if __name__ == "__main__":
    unittest.main()
