#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from qualtran.bloqs.factoring.rsa.rsa_phase_estimate import _rsa_pe, _rsa_pe_small


def test_rsa_pe(bloq_autotester):
    bloq_autotester(_rsa_pe)


def test_rsa_pe_small(bloq_autotester):
    bloq_autotester(_rsa_pe_small)
