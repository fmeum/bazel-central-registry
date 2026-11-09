#!/usr/bin/env python3
#
# Copyright 2026 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import io
import os
import pathlib
import subprocess
import tempfile
import unittest

from unittest.mock import patch

from tools.toolchain_registrations import (
    MAX_ANNOTATIONS,
    AnalysisError,
    Call,
    CallResult,
    ModuleResult,
    RegistrationAnalyzer,
    added_module_versions,
    analyze_module,
    canonicalize_pattern,
    describe_bazel_error,
    find_registration_calls,
    github_annotations,
    github_summary,
    main,
    owning_module,
)


def rule(name, rule_class, **attrs):
    return {
        "name": name,
        "ruleClass": rule_class,
        "attribute": [{"name": k, "type": "LABEL", "stringValue": v} for k, v in attrs.items()],
    }


class FakeQuery:
    def __init__(self, results):
        self.results = results
        self.expressions = []

    def __call__(self, expression):
        self.expressions.append(expression)
        return self.results[expression]


class FindRegistrationCallsTest(unittest.TestCase):
    def test_returns_non_dev_calls(self):
        calls = find_registration_calls(
            """module(name = "foo")

register_toolchains("//:a", "@hub//:all")
register_toolchains(
    "//:b",
    dev_dependency = False,
)
register_toolchains("//:c", dev_dependency = True)
"""
        )
        self.assertEqual(calls, [Call(3, 3, ["//:a", "@hub//:all"], []), Call(4, 7, ["//:b"], [])])

    def test_reports_what_cant_be_analyzed_statically(self):
        calls = find_registration_calls(
            """register_toolchains("//:a", "@" + NAME + "//:all")
[register_toolchains(name) for name in NAMES]
register_toolchains("//:b", dev_dependency = IS_DEV)
register_toolchains(name, dev_dependency = True)
"""
        )
        self.assertEqual(
            calls,
            [
                Call(1, 1, ["//:a"], ["'@' + NAME + '//:all' isn't a string literal"]),
                Call(2, 2, [], ["name isn't a string literal"]),
                Call(3, 3, [], ["dev_dependency isn't a literal"]),
            ],
        )


class CanonicalizePatternTest(unittest.TestCase):
    def test_own_repo(self):
        self.assertEqual(canonicalize_pattern("//foo:all", "m+", {}), "@@m+//foo:all")

    def test_apparent_repo(self):
        self.assertEqual(canonicalize_pattern("@hub//foo/...", "m+", {"hub": "m++ext+hub"}), "@@m++ext+hub//foo/...")

    def test_canonical_repo(self):
        self.assertEqual(canonicalize_pattern("@@other+//:all", "m+", {}), "@@other+//:all")

    def test_unknown_repo(self):
        with self.assertRaises(AnalysisError):
            canonicalize_pattern("@unknown//:all", "m+", {})


class OwningModuleTest(unittest.TestCase):
    def test_owning_module(self):
        for label, module in [
            ("@@rules_foo+//:type", "rules_foo"),
            ("@@rules_foo+1.2.3//:type", "rules_foo"),
            ("@@rules_foo++ext+hub//:type", "rules_foo"),
            ("@@rules_foo+1.2.3+ext+hub//:type", "rules_foo"),
            ("@@bazel_tools//tools/cpp:toolchain_type", "bazel_tools"),
        ]:
            self.assertEqual(owning_module(label), module, label)


class RegistrationAnalyzerTest(unittest.TestCase):
    def test_wildcard_ignores_aliases(self):
        query = FakeQuery(
            {
                'kind("^(toolchain) rule$", "@@m+//:all")': [
                    rule("@@m+//:t", "toolchain", toolchain_type="@@m+//:type")
                ],
                '"@@m+//:type"': [rule("@@m+//:type", "toolchain_type")],
            }
        )
        self.assertEqual(RegistrationAnalyzer(query).toolchain_types("@@m+//:all"), {"@@m+//:type"})

    def test_follows_alias_named_explicitly(self):
        query = FakeQuery(
            {
                'kind("^(toolchain|alias) rule$", "@@m++ext+hub//:alias")': [
                    rule("@@m++ext+hub//:alias", "alias", actual="@@m+//:t")
                ],
                '"@@m+//:t"': [rule("@@m+//:t", "toolchain", toolchain_type="@@m+//:type")],
                '"@@m+//:type"': [rule("@@m+//:type", "toolchain_type")],
            }
        )
        self.assertEqual(RegistrationAnalyzer(query).toolchain_types("@@m++ext+hub//:alias"), {"@@m+//:type"})

    def test_follows_toolchain_type_alias(self):
        query = FakeQuery(
            {
                'kind("^(toolchain|alias) rule$", "@@m+//:t")': [
                    rule("@@m+//:t", "toolchain", toolchain_type="@@bazel_tools//:type")
                ],
                '"@@bazel_tools//:type"': [rule("@@bazel_tools//:type", "alias", actual="@@rules_foo+//:type")],
                '"@@rules_foo+//:type"': [rule("@@rules_foo+//:type", "toolchain_type")],
            }
        )
        self.assertEqual(RegistrationAnalyzer(query).toolchain_types("@@m+//:t"), {"@@rules_foo+//:type"})

    def test_queries_each_label_once(self):
        query = FakeQuery(
            {
                'kind("^(toolchain) rule$", "@@m+//:all")': [
                    rule("@@m+//:t1", "toolchain", toolchain_type="@@m+//:type"),
                    rule("@@m+//:t2", "toolchain", toolchain_type="@@m+//:type"),
                ],
                'kind("^(toolchain|alias) rule$", "@@m+//:t1")': [
                    rule("@@m+//:t1", "toolchain", toolchain_type="@@m+//:type")
                ],
                '"@@m+//:type"': [rule("@@m+//:type", "toolchain_type")],
            }
        )
        analyzer = RegistrationAnalyzer(query)
        analyzer.toolchain_types("@@m+//:all")
        analyzer.toolchain_types("@@m+//:t1")
        self.assertEqual(
            query.expressions,
            [
                'kind("^(toolchain) rule$", "@@m+//:all")',
                '"@@m+//:type"',
                'kind("^(toolchain|alias) rule$", "@@m+//:t1")',
            ],
        )

    def test_alias_with_select(self):
        alias = {
            "name": "@@m+//:alias",
            "ruleClass": "alias",
            "attribute": [{"name": "actual", "type": "LABEL", "selectorList": {}}],
        }
        query = FakeQuery(
            {
                'kind("^(toolchain|alias) rule$", "@@m+//:t")': [
                    rule("@@m+//:t", "toolchain", toolchain_type="@@m+//:alias")
                ],
                '"@@m+//:alias"': [alias],
            }
        )
        with self.assertRaisesRegex(AnalysisError, "doesn't have a constant value"):
            RegistrationAnalyzer(query).toolchain_types("@@m+//:t")

    def test_alias_cycle(self):
        query = FakeQuery(
            {
                'kind("^(toolchain|alias) rule$", "@@m+//:t")': [
                    rule("@@m+//:t", "toolchain", toolchain_type="@@m+//:a")
                ],
                '"@@m+//:a"': [rule("@@m+//:a", "alias", actual="@@m+//:b")],
                '"@@m+//:b"': [rule("@@m+//:b", "alias", actual="@@m+//:a")],
            }
        )
        with self.assertRaisesRegex(AnalysisError, "alias cycle"):
            RegistrationAnalyzer(query).toolchain_types("@@m+//:t")


class DescribeBazelErrorTest(unittest.TestCase):
    def test_prefers_starlark_root_cause_with_location(self):
        stderr = """INFO: Invocation ID: 123
ERROR: Traceback (most recent call last):
\tFile "/ob/external/m+/toolchains.bzl", line 7, column 26, in <toplevel>
\t\tfoo_toolchain = rule(
Error in rule: rule() got unexpected keyword argument 'incompatible_use_toolchain_transition'
ERROR: error loading package '@@m+//toolchains': initialization of module 'toolchains.bzl' failed
"""
        self.assertEqual(
            describe_bazel_error(stderr, 7, ["/ob/external/"]),
            "m+/toolchains.bzl:7:26: Error in rule: rule() got unexpected keyword argument"
            " 'incompatible_use_toolchain_transition'",
        )

    def test_falls_back_to_error_lines(self):
        stderr = """ERROR: /ob/external/m+/defs.bzl:26:12: The CcInfo symbol has been removed
ERROR: /ob/external/m+/defs.bzl:26:12: The CcInfo symbol has been removed
ERROR: Target parsing failed
"""
        self.assertEqual(
            describe_bazel_error(stderr, 7, ["/ob/external/"]),
            "m+/defs.bzl:26:12: The CcInfo symbol has been removed\nTarget parsing failed",
        )

    def test_joins_multiline_messages_and_skips_builtins(self):
        stderr = """ERROR: Traceback (most recent call last):
\tFile "/ob/external/m+/BUILD", line 4, column 8, in <toplevel>
\t\tcc_test(
\tFile "/virtual_builtins_bzl/bazel/exports.bzl", line 40, column 9, in _removed_rule_failure
Error in fail:
         This rule has been removed from Bazel.
         Please add a `load()` statement for it.
Loading: 1 packages loaded
ERROR: error loading package '@@m+//': compilation failed
"""
        self.assertEqual(
            describe_bazel_error(stderr, 7, ["/ob/external/"]),
            "m+/BUILD:4:8: Error in fail: This rule has been removed from Bazel. Please add a `load()` statement for it.",
        )

    def test_joins_multiline_error_lines(self):
        stderr = """ERROR: m+/defs.bzl:26:12: The CcInfo symbol has been removed, add the following to your BUILD/bzl file:
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
    currently loading: @@m+//
"""
        self.assertEqual(
            describe_bazel_error(stderr, 7, []),
            "m+/defs.bzl:26:12: The CcInfo symbol has been removed, add the following to your BUILD/bzl file:"
            ' load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")',
        )

    def test_falls_back_to_exit_code_and_tail(self):
        self.assertEqual(
            describe_bazel_error("\nsomething\nwent wrong\n", 37, []), "exit code 37\nsomething\nwent wrong"
        )


class AnalyzeModuleTest(unittest.TestCase):
    def analyze(self, module_dot_bazel):
        with tempfile.TemporaryDirectory() as root:
            module_dir = pathlib.Path(root) / "modules" / "foo" / "1.0"
            module_dir.mkdir(parents=True)
            (module_dir / "MODULE.bazel").write_text(module_dot_bazel)
            return analyze_module(root, "foo", "1.0")

    @patch("tools.toolchain_registrations.Bazel")
    def test_no_bazel_without_non_dev_registrations(self, bazel):
        result = self.analyze('register_toolchains("//:a", dev_dependency = True)\n')
        self.assertEqual(result, ModuleResult("foo", "1.0", "modules/foo/1.0/MODULE.bazel", []))
        self.assertEqual(result.status, "no non-dev registrations")
        bazel.assert_not_called()

    @patch("tools.toolchain_registrations.Bazel")
    def test_no_bazel_without_string_literals(self, bazel):
        result = self.analyze("register_toolchains(NAME)\n")
        self.assertEqual(result.calls[0].problems, [(None, "NAME isn't a string literal")])
        self.assertEqual(result.status, "couldn't be fully analyzed")
        bazel.assert_not_called()

    @patch("tools.toolchain_registrations.Bazel")
    def test_unparsable_file(self, bazel):
        result = self.analyze('\nregister_toolchains("//:a"\n')
        self.assertEqual(result.calls[0].call.line, 2)
        self.assertIn("can't parse the file", result.calls[0].problems[0][1])
        bazel.assert_not_called()

    @patch("tools.toolchain_registrations.Bazel")
    def test_findings_and_problems(self, bazel):
        bazel.return_value.repo_mapping.side_effect = lambda repo: {"": {"foo": "foo+"}, "foo+": {"hub": "foo++e+hub"}}[
            repo
        ]
        bazel.return_value.query_rules.side_effect = FakeQuery(
            {
                'kind("^(toolchain) rule$", "@@foo+//:all")': [
                    rule("@@foo+//:t", "toolchain", toolchain_type="@@bazel_tools//tools/cpp:toolchain_type"),
                    rule("@@foo+//:u", "toolchain", toolchain_type="@@foo+//:type"),
                ],
                '"@@bazel_tools//tools/cpp:toolchain_type"': [
                    rule("@@bazel_tools//tools/cpp:toolchain_type", "toolchain_type")
                ],
                '"@@foo+//:type"': [rule("@@foo+//:type", "toolchain_type")],
                'kind("^(toolchain) rule$", "@@foo++e+hub//:all")': [
                    rule("@@foo++e+hub//:t", "toolchain", toolchain_type="@@foo+//:type")
                ],
            }
        )
        result = self.analyze('register_toolchains("//:all", "@hub//:all", "@unknown//:all")\n')
        self.assertEqual(
            result.calls,
            [
                CallResult(
                    Call(1, 1, ["//:all", "@hub//:all", "@unknown//:all"], []),
                    {"//:all": ["@@bazel_tools//tools/cpp:toolchain_type"]},
                    [("@unknown//:all", "can't resolve the repo of @unknown//:all in the module's repo mapping")],
                )
            ],
        )
        self.assertEqual(result.status, "registers toolchains for types it doesn't own, couldn't be fully analyzed")
        bazel.return_value.shutdown.assert_called_once()

    @patch("tools.toolchain_registrations.Bazel")
    def test_setup_failure(self, bazel):
        bazel.return_value.repo_mapping.side_effect = AnalysisError("boom")
        result = self.analyze('register_toolchains("//:a")\nregister_toolchains("//:b")\n')
        self.assertEqual([call.problems for call in result.calls], [[(None, "boom")], [(None, "boom")]])


class AddedModuleVersionsTest(unittest.TestCase):
    def test_added_module_versions(self):
        with tempfile.TemporaryDirectory() as root:

            def git(*args):
                subprocess.run(
                    ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
                    cwd=root,
                    check=True,
                    capture_output=True,
                )

            def write(path):
                (pathlib.Path(root) / path).parent.mkdir(parents=True, exist_ok=True)
                (pathlib.Path(root) / path).write_text(path)

            git("init", "-q")
            write("modules/old/1.0/MODULE.bazel")
            git("add", ".")
            git("commit", "-q", "-m", "base")
            write("modules/old/1.0/MODULE.bazel.new")
            write("modules/old/2.0/MODULE.bazel")
            write("modules/new/1.0/MODULE.bazel")
            write("modules/new/1.0/overlay/MODULE.bazel")
            git("add", ".")
            git("commit", "-q", "-m", "head")
            self.assertEqual(sorted(added_module_versions(root, "HEAD^")), [("new", "1.0"), ("old", "2.0")])


class GitHubReportTest(unittest.TestCase):
    def result(self, calls=1, problems=()):
        call_results = [
            CallResult(Call(i, i + 1, ["//:all"], []), {"//:all": ["@@bazel_tools//tools/cpp:toolchain_type"]}, [])
            for i in range(calls)
        ]
        call_results += [CallResult(Call(100, 100, [], []), {}, list(problems))]
        return ModuleResult("foo", "1.0", "modules/foo/1.0/MODULE.bazel", call_results)

    def test_annotations(self):
        result = self.result(
            problems=[("@hub//:all", "line 1\nline 2: 100%"), ("@other//:all", "line 1\nline 2: 100%")]
        )
        foreign, incomplete = github_annotations([result], "9.3.2")
        self.assertTrue(
            foreign.startswith(
                "::warning file=modules/foo/1.0/MODULE.bazel,line=0,endLine=1,"
                "title=foo registers toolchains for types it doesn't own::"
                "These registrations apply to all dependents of foo"
            ),
            foreign,
        )
        self.assertIn("%0A- //:all: @@bazel_tools//tools/cpp:toolchain_type%0A", foreign)
        self.assertEqual(
            incomplete,
            "::warning file=modules/foo/1.0/MODULE.bazel,line=100,endLine=100,"
            "title=Couldn't analyze toolchain registrations of foo::"
            "Couldn't determine which toolchain types this registers:%0A"
            "- @hub//:all, @other//:all: line 1%0Aline 2: 100%25",
        )

    def test_annotation_limit(self):
        annotations = github_annotations([self.result(calls=MAX_ANNOTATIONS + 2)], None)
        self.assertEqual(len(annotations), MAX_ANNOTATIONS)
        self.assertEqual(
            annotations[-1], "::warning::3 more toolchain registration warnings are only listed in the job summary."
        )

    def test_summary(self):
        summary = github_summary([self.result(problems=[(None, "boom")])], "9.3.2")
        self.assertIn(
            "| `foo@1.0` | registers toolchains for types it doesn't own, couldn't be fully analyzed |", summary
        )
        self.assertIn("`modules/foo/1.0/MODULE.bazel:100`\n```\nCouldn't determine", summary)
        self.assertNotIn("Analyzed with Bazel", summary.split("Couldn't determine")[1])
        self.assertIn("This PR doesn't add any module versions.", github_summary([], "9.3.2"))

    @patch("tools.toolchain_registrations.analyze_module")
    def test_main_doesnt_let_module_output_issue_commands(self, analyze_module):
        problem = (None, "\n::error::injected")
        analyze_module.return_value = self.result(calls=0, problems=[problem])
        with tempfile.TemporaryDirectory() as root:
            summary_path = pathlib.Path(root) / "summary.md"
            stdout = io.StringIO()
            with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(summary_path)}), contextlib.redirect_stdout(stdout):
                self.assertEqual(main(["--registry", root, "--check=foo@1.0", "--github"]), 0)
            lines = stdout.getvalue().splitlines()
            self.assertTrue(lines[0].startswith("::stop-commands::"))
            resume = lines.index(f"::{lines[0].removeprefix('::stop-commands::')}::")
            self.assertTrue(any("::error::injected" in line for line in lines[1:resume]))
            self.assertTrue(all("%0A::error::injected" in line for line in lines[resume + 1 :]))
            self.assertIn("## Toolchain registrations", summary_path.read_text())

    @patch("tools.toolchain_registrations.analyze_module", side_effect=RuntimeError("bug"))
    def test_main_fails_on_crash(self, _):
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--registry", root, "--check=foo@1.0"]), 1)


if __name__ == "__main__":
    unittest.main()
