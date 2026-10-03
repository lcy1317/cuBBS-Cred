// cuBBS-Cred's local integration manifest; the upstream W3C tests are unmodified.
const {execFileSync} = require('node:child_process');
const baseUrl = process.env.CUBBS_TEST_BASE_URL || 'http://127.0.0.1:40444/id';
const health = JSON.parse(execFileSync('curl', ['-fsS', `${baseUrl}/healthz`], {
  encoding: 'utf8', timeout: 5000
}));
if(health.bbs_backend_fallback || !health.bbs_backend_hardware_acceleration) {
  throw new Error('W3C GPU acceptance requires a live hardware backend, without fallback');
}
const supports = {vc: ['1.1', '2.0']};
const tags = ['bbs-2023', 'vc-api', 'vc2.0'];
module.exports = {
  settings: {enableInteropTests: true, testAllImplementations: false},
  implementations: [{
    name: 'cuBBS-Cred', implementation: 'cuBBS-Cred CUDA BBS backend',
    issuers: [{id: health.did, endpoint: `${baseUrl}/credentials/issue`,
      options: {type: 'DataIntegrityProof'}, supports, tags}],
    verifiers: [{endpoint: `${baseUrl}/credentials/verify`, supports, tags}],
    vcHolders: [{endpoint: `${baseUrl}/credentials/derive`, supports,
      tags: ['vcHolder', ...tags]}]
  }]
};
