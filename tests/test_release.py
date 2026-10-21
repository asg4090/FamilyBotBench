"""Release interface and credential isolation regressions (no network calls)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
import tempfile
import os
import yaml
from provider_client import load_llm_settings, UnifiedLLMClient, load_local_env
from release_support import load_yaml, parse_release_args, redact_error
import argparse

class ReleaseTests(unittest.TestCase):
    def test_drift_model_pool_uses_environment_models(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'RedditBotSim'))
        from persona_social.drift_controller import filtered_model_pool
        with patch.dict(os.environ,{'OPENAI_MODEL':'test-openai-model','GEMINI_MODEL':'test-gemini-model'},clear=True):
            pool=filtered_model_pool(['gemini'])
            self.assertTrue(pool)
            self.assertEqual({item['model'] for item in pool},{'test-gemini-model'})
            self.assertEqual({item['provider'] for item in pool},{'gemini'})

    def test_credentials_are_provider_specific(self):
        with patch.dict(os.environ,{'OPENAI_API_KEY':'test-only-openai'},clear=True):
            self.assertEqual(load_llm_settings().api_key,'test-only-openai')
            for provider in ['gemini','anthropic']:
                with self.assertRaisesRegex(RuntimeError,'Missing'):
                    load_llm_settings(provider_override=provider)

    def test_no_implicit_env_file_loading(self):
        with patch.dict(os.environ,{},clear=True), tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'.env.local';p.write_text('OPENAI_API_KEY=test-only-file-value\n')
            load_local_env(p)
            with self.assertRaisesRegex(RuntimeError,'Missing'):
                load_llm_settings()

    def test_config_cannot_supply_literal_credentials(self):
        with patch.dict(os.environ,{'OPENAI_API_KEY':'test-only-value'},clear=True):
            with self.assertRaisesRegex(ValueError,'Literal credentials'):
                load_llm_settings({'llm':{'api_key':'test-only-literal'}})

    def test_secret_not_in_repr_or_errors(self):
        value='test-only-secret-with-length'
        with patch.dict(os.environ,{'OPENAI_API_KEY':value},clear=True):
            settings=load_llm_settings()
            self.assertNotIn(value,repr(settings))
            def failure(**kwargs):raise RuntimeError('credential='+value)
            raw=NS(responses=NS(create=failure))
            client=UnifiedLLMClient({'openai':raw},{'openai':settings},settings)
            with self.assertRaisesRegex(RuntimeError,'REDACTED') as error:
                client.generate_text('hello')
            self.assertNotIn(value,str(error.exception))
            self.assertNotIn(value,redact_error('Authorization: Bearer '+value))

    def test_mocked_provider_responses(self):
        for provider in ['openai','gemini','anthropic']:
            with self.subTest(provider=provider),patch.dict(os.environ,{provider.upper()+'_API_KEY':'test-only-value'},clear=True):
                settings=load_llm_settings(provider_override=provider)
                seen=[]
                def response(**kwargs):
                    seen.append(kwargs)
                    return NS(output_text='mock response',text='mock response',content=[NS(text='mock response')])
                raw=NS(responses=NS(create=response),models=NS(generate_content=response),messages=NS(create=response))
                client=UnifiedLLMClient({provider:raw},{provider:settings},settings)
                self.assertEqual(client.generate_text('test prompt'),'mock response')
                self.assertEqual(len(seen),1)
                self.assertEqual(seen[0]['model'],settings.model)
                with self.assertRaisesRegex(RuntimeError,'not configured'):
                    client.generate_text('test',settings_override={'provider':'missing-provider'})

    def test_config_paths_are_file_relative_and_cli_wins(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp).resolve();config=base/'config/release.yaml';config.parent.mkdir()
            config.write_text(yaml.safe_dump({'paths':{'original_root':'../raw','runtime_config':'runtime.yaml'},'pipelines':{'test':{'output_dir':'../results'}}}))
            self.assertEqual(load_yaml(config)['paths']['original_root'],str(base/'raw'))
            namespace={}
            parser=argparse.ArgumentParser();parser.add_argument('--output-dir',default='default')
            with patch.object(sys,'argv',['test','--config',str(config),'--output-dir',str(base/'chosen')]):
                args=parse_release_args(parser,namespace,'test')
            self.assertEqual(args.output_dir,str(base/'chosen'))
            self.assertEqual(namespace['DATA_ROOT'],base/'raw')
            self.assertEqual(args.runtime_config,str(config.parent/'runtime.yaml'))

if __name__=='__main__':unittest.main()
