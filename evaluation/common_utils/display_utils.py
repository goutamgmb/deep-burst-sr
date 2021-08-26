# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def generate_formatted_report(scores_all, table_name=''):
    name_width = max([len(d) for d in scores_all.keys()] + [len(table_name)]) + 5
    min_score_width = 10

    report_text = '\n{label: <{width}} |'.format(label=table_name, width=name_width)

    metrics = list(scores_all.values())[0].keys()
    score_widths = [max(min_score_width, len(k) + 3) for k in metrics]

    for s, s_w in zip(metrics, score_widths):
        report_text = '{prev} {s: <{width}} |'.format(prev=report_text, s=s, width=s_w)

    report_text = '{prev}\n'.format(prev=report_text)

    for network_name, network_scores in scores_all.items():
        # display name
        report_text = '{prev}{method: <{width}} |'.format(prev=report_text, method=network_name,
                                                           width=name_width)
        for (score_type, score_value), s_w in zip(network_scores.items(), score_widths):
            report_text = '{prev} {score: <{width}} |'.format(prev=report_text,
                                                              score='{:0.3f}'.format(score_value),
                                                              width=s_w)
        report_text = '{prev}\n'.format(prev=report_text)

    return report_text
