import type { BrainInfo } from '../../types';

interface Props {
  brains: BrainInfo[];
  activeBrainId: number | null;
  trainingBrainId?: number | null;
  onSelect: (id: number) => void;
  onDelete: (id: number) => void;
  onCreateClick: () => void;
}

export function Sidebar({ brains, activeBrainId, trainingBrainId, onSelect, onDelete, onCreateClick }: Props) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">Brains</div>
      <div className="brain-list">
        {brains.map((b) => {
          const isActive = b.id === activeBrainId;
          const isTraining = b.id === trainingBrainId;
          const cls = `brain-card${isActive ? ' active' : ''}${isTraining ? ' training' : ''}`;
          return (
            <div
              key={b.id}
              className={cls}
              onClick={() => onSelect(b.id)}
            >
              <button
                className="brain-card-delete"
                onClick={(e) => { e.stopPropagation(); onDelete(b.id); }}
                title="Delete"
              >
                x
              </button>
              <div className="brain-card-name">{b.name}</div>
              <div className="brain-card-meta">
                {b.probe ? `${b.probe.num_neurons.toLocaleString()} neurons` : 'Loading...'}
                {b.probe?.is_cow_clone && ' (COW)'}
              </div>
            </div>
          );
        })}
        {brains.length === 0 && (
          <div style={{ padding: 16, fontSize: 12, color: 'var(--text-muted)' }}>
            No brains yet
          </div>
        )}
      </div>
      <div className="sidebar-actions">
        <button className="btn btn-primary btn-block" onClick={onCreateClick}>
          + New Brain
        </button>
      </div>
    </aside>
  );
}
