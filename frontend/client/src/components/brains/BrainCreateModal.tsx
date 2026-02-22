import { useState } from 'react';
import type { BrainCreate } from '../../types';

interface Props {
  onClose: () => void;
  onCreate: (data: BrainCreate) => void;
}

const SIZES = ['Tiny (100 neurons)', 'Small (1K neurons)', 'Medium (10K neurons)', 'Large (100K neurons)'];
const TASKS = ['Classification', 'Regression', 'Pattern Matching', 'Sequence', 'Association'];

export function BrainCreateModal({ onClose, onCreate }: Props) {
  const [name, setName] = useState('brain');
  const [size, setSize] = useState(1);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [task, setTask] = useState<number | undefined>(undefined);
  const [numInputs, setNumInputs] = useState<number | undefined>(undefined);
  const [numOutputs, setNumOutputs] = useState<number | undefined>(undefined);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const data: BrainCreate = { name, size };
    if (showAdvanced) {
      if (task !== undefined) data.task = task;
      if (numInputs !== undefined) data.num_inputs = numInputs;
      if (numOutputs !== undefined) data.num_outputs = numOutputs;
    }
    onCreate(data);
  };

  const isAutoDetect = !showAdvanced || (task === undefined && numInputs === undefined && numOutputs === undefined);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-title">Create Brain</div>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Name</label>
            <input value={name} onChange={(e) => setName(e.target.value)} required />
          </div>
          <div className="form-group">
            <label>Size</label>
            <select value={size} onChange={(e) => setSize(Number(e.target.value))}>
              {SIZES.map((s, i) => <option key={i} value={i}>{s}</option>)}
            </select>
          </div>
          {isAutoDetect && (
            <div style={{ padding: '8px 12px', marginBottom: 12, background: 'rgba(99,102,241,0.1)', borderRadius: 6, fontSize: 12, color: 'var(--text-muted)' }}>
              Conversational brain — ready to chat, learn, and grow
            </div>
          )}
          <div style={{ marginBottom: 12 }}>
            <button
              type="button"
              style={{
                background: 'none', border: 'none', color: 'var(--accent)',
                fontSize: 12, cursor: 'pointer', padding: 0,
                display: 'flex', alignItems: 'center', gap: 4,
              }}
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              <span style={{ transform: showAdvanced ? 'rotate(90deg)' : 'rotate(0)', transition: 'transform 0.2s', display: 'inline-block' }}>&#9654;</span>
              Advanced
            </button>
          </div>
          {showAdvanced && (
            <>
              <div className="form-group">
                <label>Task</label>
                <select
                  value={task ?? ''}
                  onChange={(e) => setTask(e.target.value === '' ? undefined : Number(e.target.value))}
                >
                  <option value="">Auto-detect</option>
                  {TASKS.map((t, i) => <option key={i} value={i}>{t}</option>)}
                </select>
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label>Inputs</label>
                  <input
                    type="number" min={1} placeholder="Auto"
                    value={numInputs ?? ''}
                    onChange={(e) => setNumInputs(e.target.value === '' ? undefined : Number(e.target.value))}
                  />
                </div>
                <div className="form-group">
                  <label>Outputs</label>
                  <input
                    type="number" min={1} placeholder="Auto"
                    value={numOutputs ?? ''}
                    onChange={(e) => setNumOutputs(e.target.value === '' ? undefined : Number(e.target.value))}
                  />
                </div>
              </div>
            </>
          )}
          {size >= 2 && (
            <div style={{ padding: '8px 12px', marginBottom: 12, background: 'rgba(245,158,11,0.15)', borderRadius: 6, fontSize: 12, color: 'var(--warning)' }}>
              Warning: {size === 2 ? 'Medium' : 'Large'} brains allocate {size === 2 ? '~50MB' : '~500MB'} RAM
            </div>
          )}
          <div className="modal-actions">
            <button type="button" className="btn btn-outline" onClick={onClose}>Cancel</button>
            <button type="submit" className="btn btn-primary">Create</button>
          </div>
        </form>
      </div>
    </div>
  );
}
